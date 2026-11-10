// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "Utils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void invalidate(__gm__ T *ptr) {
  __asm__ __volatile__("");
  // Same A2 instruction used by AscendC DataCacheCleanAndInvalid, without
  // kernel_operator.h's g_tilingKey global (plain/mix bitcode collision).
  dcci(reinterpret_cast<__gm__ uint64_t *>(ptr),
       cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
  __asm__ __volatile__("");
}

__aiv__ __attribute__((always_inline)) int64_t await_epoch(
    __gm__ int64_t *flag, int64_t expected) {
  // A fixed spin count is not a distributed deadline: another rank may still
  // be loading its first kernel. External execution guards own wall timeouts.
  int64_t observed;
  while (true) {
    invalidate(flag);
    observed = *reinterpret_cast<volatile __gm__ int64_t *>(flag);
    if (observed == expected) return observed;
  }
}

extern "C" __aiv__ __attribute__((always_inline)) void _mlir_ciface_moe_scatter_rows(
    memref_t<__ubuf__ float, 2> *values,
    memref_t<__ubuf__ int32_t, 1> *routes,
    memref_t<__gm__ float, 1> *output,
    memref_t<__gm__ int64_t, 1> *status,
    int32_t valid, int32_t hidden, int32_t column, int32_t route_limit,
    int32_t tile_rows, int32_t tile_cols, int32_t combine_topk) {
  pipe_barrier(PIPE_ALL);
  auto control = status->aligned + status->offset;
  control[2] = 0;
  invalidate(control);
  if (values->sizes[0] != tile_rows || values->sizes[1] != tile_cols ||
      values->strides[1] != 1 || routes->sizes[0] != tile_rows ||
      tile_cols % 8 != 0 || valid < 0 || valid > tile_rows) {
    *reinterpret_cast<volatile __gm__ int64_t *>(control + 2) = 31;
    invalidate(control);
    return;
  }
  // Atomic DMA uses only valid expert rows. The generic masked atomic path
  // still issued zero-valued stores for every padded row, serializing MTE3.
  if (combine_topk > 0) {
    set_atomic_f32();
    set_atomic_add();
  }
  for (int32_t row = 0; row < valid; ++row) {
    int32_t route = routes->aligned[routes->offset + row * routes->strides[0]];
    if (route < 0 || route >= route_limit) {
      *reinterpret_cast<volatile __gm__ int64_t *>(control + 2) = 32;
      invalidate(control);
      continue;
    }
    int32_t output_row = combine_topk > 0 ? route / combine_topk : route;
    copy_ubuf_to_gm(output->aligned + output->offset + static_cast<int64_t>(output_row) * hidden + column,
                    values->aligned + values->offset + row * values->strides[0],
                    0, 1, tile_cols / 8, 0, 0);
  }
  if (combine_topk > 0) {
    pipe_barrier(PIPE_MTE3);
    set_atomic_none();
  }
  pipe_barrier(PIPE_ALL);
}

extern "C" __aiv__ __attribute__((always_inline)) void _mlir_ciface_moe_read_epoch(
    memref_t<__gm__ int64_t, 1> *epoch,
    memref_t<__ubuf__ int64_t, 1> *result) {
  pipe_barrier(PIPE_ALL);
  auto src = epoch->aligned + epoch->offset;
  invalidate(src);
  result->aligned[result->offset] =
      *reinterpret_cast<volatile __gm__ int64_t *>(src);
  pipe_barrier(PIPE_ALL);
}

extern "C" __aiv__ __attribute__((always_inline)) void _mlir_ciface_moe_begin(
    memref_t<__gm__ int64_t, 1> *epoch_ptr,
    memref_t<__gm__ int64_t, 1> *free_flags, int32_t world) {
  pipe_barrier(PIPE_ALL);
  auto state = epoch_ptr->aligned + epoch_ptr->offset;
  invalidate(state);
  int64_t epoch = *reinterpret_cast<volatile __gm__ int64_t *>(state);
  for (int32_t peer = 0; peer < world; ++peer) {
    await_epoch(free_flags->aligned + free_flags->offset + peer * 16, epoch);
  }
  // Keep the read/modify/write in one scalar cache domain. Returning through
  // a UB tensor and then an MTE store left Epoch unchanged in the regression.
  *reinterpret_cast<volatile __gm__ int64_t *>(state) = epoch + 1;
  invalidate(state);
  pipe_barrier(PIPE_ALL);
}

extern "C" __aiv__ __attribute__((always_inline)) void _mlir_ciface_moe_wait_many(
    memref_t<__gm__ int64_t, 1> *flags, int64_t epoch, int32_t count,
    memref_t<__ubuf__ int64_t, 1> *completion) {
  pipe_barrier(PIPE_ALL);
  for (int32_t peer = 0; peer < count; ++peer)
    await_epoch(flags->aligned + flags->offset + peer * 16, epoch);
  completion->aligned[completion->offset] = epoch;
  pipe_barrier(PIPE_ALL);
}

extern "C" __aiv__ __attribute__((always_inline)) void _mlir_ciface_moe_signal(
    int64_t epoch, memref_t<__gm__ int64_t, 1> *signal,
    memref_t<__ubuf__ int64_t, 1> *completion) {
  pipe_barrier(PIPE_ALL);
  auto flag = signal->aligned + signal->offset;
  *reinterpret_cast<volatile __gm__ int64_t *>(flag) = epoch;
  invalidate(flag);
  completion->aligned[completion->offset] = epoch;
  pipe_barrier(PIPE_ALL);
}

extern "C" __aiv__ __attribute__((always_inline)) void _mlir_ciface_moe_wait(
    memref_t<__gm__ int64_t, 1> *signal, int64_t expected,
    memref_t<__ubuf__ int64_t, 1> *observed) {
  pipe_barrier(PIPE_ALL);
  observed->aligned[observed->offset] =
      await_epoch(signal->aligned + signal->offset, expected);
  pipe_barrier(PIPE_ALL);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void direct_invalidate(__gm__ T *ptr) {
  dcci(reinterpret_cast<__gm__ uint64_t *>(ptr),
       cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
}

__aiv__ __attribute__((always_inline)) int64_t direct_await_epoch(__gm__ int64_t *ptr, int64_t epoch) {
  while (true) {
    direct_invalidate(ptr);
    int64_t observed = *reinterpret_cast<volatile __gm__ int64_t *>(ptr);
    if (observed == epoch) return observed;
  }
}

extern "C" __aiv__ __attribute__((always_inline)) void _mlir_ciface_direct_dispatch(
    memref_t<__gm__ bfloat16_t, 1> *input,
    memref_t<__gm__ int32_t, 1> *tokens,
    memref_t<__gm__ int32_t, 1> *experts,
    memref_t<__gm__ float, 1> *weights,
    memref_t<__gm__ int32_t, 1> *packet_count,
    memref_t<__gm__ int32_t, 1> *source_expert_counts,
    memref_t<__gm__ int64_t, 1> *published,
    memref_t<__gm__ bfloat16_t, 1> *packed,
    memref_t<__gm__ int32_t, 1> *headers,
    memref_t<__gm__ int64_t, 1> *states,
    int64_t epoch, int32_t source, int32_t rank, int32_t world,
    int32_t capacity, int32_t hidden, int32_t local_experts,
    int32_t topk, int32_t padded_topk, int32_t block_rows,
    memref_t<__gm__ int64_t, 1> *metadata_ready,
    memref_t<__gm__ int64_t, 1> *merged_states,
    memref_t<__gm__ int64_t, 1> *task_meta,
    int32_t batch_rows, int32_t no_input_padding,
    int32_t shard, int32_t shards,
    memref_t<__ubuf__ int32_t, 1> *scratch,
    memref_t<__ubuf__ int64_t, 1> *completion) {
  auto counts = scratch->aligned + scratch->offset;
  auto totals = counts + ((local_experts + 7) / 8) * 8;
  int32_t aligned_experts = ((local_experts+7)/8)*8;
  auto base_cache = reinterpret_cast<__ubuf__ int64_t *>(totals+aligned_experts);
  auto merged_totals = reinterpret_cast<__ubuf__ int32_t *>(base_cache+aligned_experts);
  auto token_buf = totals + aligned_experts + 3*aligned_experts;
  auto expert_buf = token_buf + 32;
  auto weight_buf = expert_buf + 32 * padded_topk;
  auto row_buf = reinterpret_cast<__ubuf__ bfloat16_t *>(weight_buf + 32 * padded_topk);
  auto header_buf = reinterpret_cast<__ubuf__ int32_t *>(row_buf + batch_rows * hidden);
  auto state = states->aligned + states->offset;
  auto dst = packed->aligned + packed->offset;
  auto header = headers->aligned + headers->offset;
  direct_await_epoch(published->aligned + published->offset, epoch);
  pipe_barrier(PIPE_ALL);
  copy_gm_to_ubuf(token_buf, packet_count->aligned + packet_count->offset, 0, 1, 1, 0, 0);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
  int32_t packets = token_buf[0];
  bool failed = packets < 0 || packets > capacity;
  {
    copy_gm_to_ubuf(counts, source_expert_counts->aligned + source_expert_counts->offset,
                    0, 1, (local_experts + 7) / 8, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
  }
  int32_t first_expert = local_experts * shard / shards;
  int32_t end_expert = local_experts * (shard + 1) / shards;
  if (shard == 0) {
    int64_t base = static_cast<int64_t>(source) * (capacity * topk + local_experts * block_rows);
    for (int32_t e = 0; e < local_experts; ++e) {
      {
        auto table=reinterpret_cast<__ubuf__ int64_t *>(row_buf);
        for (int32_t j=0;j<16;++j)table[e*16+j]=0;
        table[e*16+1]=base;
        table[e*16+2]=counts[e];
      }
      totals[e] = counts[e];
      base += ((counts[e] + block_rows - 1) / block_rows) * block_rows;
      counts[e] = 0;
    }
    {
      set_flag(PIPE_S,PIPE_MTE3,EVENT_ID1);
      wait_flag(PIPE_S,PIPE_MTE3,EVENT_ID1);
      copy_ubuf_to_gm(state+source*16,reinterpret_cast<__ubuf__ int64_t *>(row_buf),
                      0,local_experts,4,0,(world-1)*4);
      set_flag(PIPE_MTE3,PIPE_S,EVENT_ID2);
      wait_flag(PIPE_MTE3,PIPE_S,EVENT_ID2);
    }
    auto meta = metadata_ready->aligned + metadata_ready->offset + source * 16;
    pipe_barrier(PIPE_ALL);
    *reinterpret_cast<volatile __gm__ int64_t *>(meta) = epoch;
    direct_invalidate(meta);
  } else {
    for (int32_t e = first_expert; e < end_expert; ++e) {
      totals[e] = counts[e];
      counts[e] = 0;
    }
  }
  direct_await_epoch(task_meta->aligned + task_meta->offset, epoch);
  auto merged = merged_states->aligned + merged_states->offset;
  auto source_snapshot = reinterpret_cast<__ubuf__ int64_t *>(row_buf);
  auto merged_snapshot = source_snapshot + local_experts*world*16;
  {
    // Reuse the input tile's scratch before the first feature DMA. Source
    // counts and merged bases are immutable once task_meta is published.
    copy_gm_to_ubuf(source_snapshot,state,0,1,local_experts*world*4,0,0);
    copy_gm_to_ubuf(merged_snapshot,merged,0,1,local_experts*4,0,0);
    set_flag(PIPE_MTE2,PIPE_S,EVENT_ID7);
    wait_flag(PIPE_MTE2,PIPE_S,EVENT_ID7);
  }
  for (int32_t e = first_expert; e < end_expert; ++e) {
    int64_t merged_base=merged_snapshot[e*16+1];
    for (int32_t source_prefix=0;source_prefix<source;++source_prefix)
      merged_base+=source_snapshot[(e*world+source_prefix)*16+2];
    base_cache[e]=merged_base;
    merged_totals[e]=merged_snapshot[e*16+2];
  }
  {
    // Finish all scalar snapshot reads before reusing its UB region for DMA.
    set_flag(PIPE_S,PIPE_MTE2,EVENT_ID7);
    wait_flag(PIPE_S,PIPE_MTE2,EVENT_ID7);
  }
  if (shards > 1) {
    int32_t routes = 0;
    for (int32_t e = first_expert; e < end_expert; ++e) routes += totals[e];
    if (routes == 0) packets = 0;
  }
  // Remember published counts once per invocation. A batch only revisits
  // experts actually touched by its routes, instead of scanning all experts.
  {
    auto destinations = reinterpret_cast<__ubuf__ int64_t *>(header_buf + batch_rows * topk * 8);
    auto source_rows = reinterpret_cast<__ubuf__ int32_t *>(destinations + batch_rows * topk);
    auto expert_ids = source_rows + batch_rows * topk;
    auto published_counts = expert_ids + batch_rows * topk;
    for (int32_t e = 0; e < local_experts; ++e) published_counts[e] = 0;
  }
  for (int32_t begin = 0; begin < packets && !failed; begin += 32) {
    copy_gm_to_ubuf(token_buf, tokens->aligned + tokens->offset + begin, 0, 1, 4, 0, 0);
    copy_gm_to_ubuf(expert_buf, experts->aligned + experts->offset + static_cast<int64_t>(begin) * padded_topk,
                    0, 1, 32 * padded_topk / 8, 0, 0);
    copy_gm_to_ubuf(weight_buf, weights->aligned + weights->offset + static_cast<int64_t>(begin) * padded_topk,
                    0, 1, 32 * padded_topk / 8, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    if (shards > 1) {
      for (int32_t row = 0; row < 32 && begin + row < packets; ++row) {
        bool needed = false;
        for (int32_t slot = 0; slot < topk; ++slot) {
          int32_t e = expert_buf[row * padded_topk + slot] - rank * local_experts;
          needed |= e >= first_expert && e < end_expert;
        }
        if (!needed) token_buf[row] = -1;
      }
    }
    for (int32_t chunk=0;chunk<32 && begin+chunk<packets;chunk+=batch_rows) {
      int32_t valid_rows = packets-begin-chunk < batch_rows ? packets-begin-chunk : batch_rows;
      if (valid_rows>32-chunk) valid_rows=32-chunk;
      bool contiguous=true;
      int32_t first_token=token_buf[chunk];
      for (int32_t j=0;j<valid_rows;++j)
        contiguous=contiguous && token_buf[chunk+j]==first_token+j && token_buf[chunk+j]>=0 && token_buf[chunk+j]<capacity;
      if (contiguous) {
        copy_gm_to_ubuf(row_buf,input->aligned+input->offset+static_cast<int64_t>(first_token)*hidden,
                       0,1,valid_rows*hidden*2/32,0,0);
      } else {
        // A missing route must not split every remaining row into tiny DMAs.
        // Keep holes in row_buf so the existing expert fanout indices stay valid.
        int32_t j=0;
        while (j<valid_rows) {
          int32_t token=token_buf[chunk+j];
          if (token==-1) { ++j; continue; }
          if (token<0 || token>=capacity) {failed=true;break;}
          int32_t run=1;
          while (j+run<valid_rows && token_buf[chunk+j+run]==token+run && token+run<capacity) ++run;
          copy_gm_to_ubuf(row_buf+j*hidden,input->aligned+input->offset+static_cast<int64_t>(token)*hidden,
                         0,1,run*hidden*2/32,0,0);
          j+=run;
        }
      }
      set_flag(PIPE_MTE2,PIPE_MTE3,EVENT_ID0);
      wait_flag(PIPE_MTE2,PIPE_MTE3,EVENT_ID0);
      {
        // Plan a whole input batch before issuing fanout. Headers stay immutable
        // until every DMA drains; publish ready counts only after that drain.
        auto destinations = reinterpret_cast<__ubuf__ int64_t *>(header_buf + batch_rows * topk * 8);
        auto source_rows = reinterpret_cast<__ubuf__ int32_t *>(destinations + batch_rows * topk);
        auto expert_ids = source_rows + batch_rows * topk;
        auto published_counts = expert_ids + batch_rows * topk;
        int32_t routes_in_batch = 0;
        for (int32_t j=0;j<valid_rows;++j) {
          int32_t token=token_buf[chunk+j];
          for (int32_t k=0;k<topk;++k) {
            int32_t id=expert_buf[(chunk+j)*padded_topk+k];
            if (token<0 || id<0) continue;
            int32_t e=id-rank*local_experts;
            if (token>=capacity || e<0 || e>=local_experts) {failed=true;continue;}
            if (e < first_expert || e >= end_expert) continue;
            int32_t q=routes_in_batch++;
            destinations[q]=base_cache[e]+counts[e]++;
            source_rows[q]=j;
            expert_ids[q]=e;
            auto head=header_buf+q*8;
            for (int32_t h=0;h<8;++h)head[h]=0;
            head[0]=(source*capacity+token)*topk+k;
            head[1]=weight_buf[(chunk+j)*padded_topk+k];
          }
        }
        set_flag(PIPE_S,PIPE_MTE3,EVENT_ID1);
        wait_flag(PIPE_S,PIPE_MTE3,EVENT_ID1);
        for (int32_t q=0;q<routes_in_batch;++q) {
          int64_t row=destinations[q];
          copy_ubuf_to_gm(header+row*8,header_buf+q*8,0,1,1,0,0);
          copy_ubuf_to_gm(dst+row*hidden,row_buf+source_rows[q]*hidden,0,1,hidden*2/32,0,0);
        }
        set_flag(PIPE_MTE3,PIPE_S,EVENT_ID2);
        wait_flag(PIPE_MTE3,PIPE_S,EVENT_ID2);
        for (int32_t q=0;q<routes_in_batch;++q) {
          int32_t e=expert_ids[q];
          if (counts[e]!=published_counts[e] &&
              (counts[e]/block_rows>published_counts[e]/block_rows || counts[e]==totals[e])) {
            auto flag=state+(e*world+source)*16;
            *reinterpret_cast<volatile __gm__ uint64_t *>(flag)=
                (static_cast<uint64_t>(epoch)<<32)|counts[e];
            direct_invalidate(flag);
            published_counts[e]=counts[e];
          }
        }
      }
    }
  }
  // Invalid M rows are independent GEMM/activation rows and are never scattered.
  // The pool reserves their addresses, so diagnostic no-padding may leave even
  // NaNs there without changing any valid row. Ready still follows real writes.
  if (!no_input_padding) {
    for (int32_t h = 0; h < hidden / 2; ++h)
      reinterpret_cast<__ubuf__ int32_t *>(row_buf)[h] = 0;
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
  }
  for (int32_t e = first_expert; e < end_expert; ++e) {
    auto flag = state + (e * world + source) * 16;
    if (!no_input_padding && source == world - 1) {
      int32_t total = merged_totals[e];
      int32_t padded = ((total + block_rows - 1) / block_rows) * block_rows;
      for (int32_t row = total; row < padded; ++row)
        copy_ubuf_to_gm(dst + (base_cache[e]+totals[e]+row-total) * hidden, row_buf, 0, 1, hidden * 2 / 32, 0, 0);
    }
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID2);
    *reinterpret_cast<volatile __gm__ uint64_t *>(flag) =
        (static_cast<uint64_t>(epoch) << 32) | 0x80000000u |
        (failed ? 0x40000000u : 0u) | counts[e];
    direct_invalidate(flag);
  }
  completion->aligned[completion->offset] = failed ? -1 : epoch;
  pipe_barrier(PIPE_ALL);
}

extern "C" __aiv__ __attribute__((always_inline)) void _mlir_ciface_scalar_plan_dma(
    memref_t<__gm__ int64_t,1> *source_states, memref_t<__gm__ int64_t,1> *merged_states,
    memref_t<__gm__ int64_t,1> *source_meta, memref_t<__gm__ int64_t,1> *task_meta,
    memref_t<__gm__ int32_t,1> *expert_counts, int64_t epoch, int32_t world, int32_t experts, int32_t bm, memref_t<__ubuf__ int32_t,1> *scratch, memref_t<__ubuf__ int32_t,1> *totals_buffer) {
  pipe_barrier(PIPE_ALL);
  auto src = source_states->aligned + source_states->offset;
  auto merged = merged_states->aligned + merged_states->offset;
  auto meta = source_meta->aligned + source_meta->offset;
  for (int32_t s=0; s<world; ++s) {
    while (true) {
      direct_invalidate(meta+s*16);
      if (*reinterpret_cast<volatile __gm__ int64_t *>(meta+s*16) == epoch) break;
    }
  }
  auto snapshot = reinterpret_cast<__ubuf__ int64_t *>(scratch->aligned + scratch->offset);
  copy_gm_to_ubuf(snapshot, src, 0, 1, experts*world*4, 0, 0);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
  auto totals = totals_buffer->aligned + totals_buffer->offset;
  int64_t base = 0;
  int32_t block_sum = 0;
  for (int32_t e=0; e<experts; ++e) {
    int32_t count = 0;
    for (int32_t s=0; s<world; ++s) {
      count += snapshot[(e*world+s)*16+2];
    }
    int32_t blocks = (count+bm-1)/bm;
    block_sum += blocks;
    {
      // This expert's source counts were consumed above. The compact output
      // lies behind all unread source rows (world>=1), so scratch is reusable.
      for (int32_t j=0;j<16;++j)snapshot[e*16+j]=0;
      snapshot[e*16+1]=base;
      snapshot[e*16+2]=count;
    }
    if (experts % 8 == 0) {
      totals[e] = count;
    } else {
      auto cnt = expert_counts->aligned + expert_counts->offset + e;
      *reinterpret_cast<volatile __gm__ int32_t *>(cnt) = count;
      dcci(reinterpret_cast<__gm__ uint64_t *>(cnt), cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    }
    base += blocks*bm;
  }
  {
    set_flag(PIPE_S,PIPE_MTE3,EVENT_ID6);
    wait_flag(PIPE_S,PIPE_MTE3,EVENT_ID6);
    copy_ubuf_to_gm(merged,snapshot,0,1,experts*4,0,0);
    set_flag(PIPE_MTE3,PIPE_S,EVENT_ID6);
    wait_flag(PIPE_MTE3,PIPE_S,EVENT_ID6);
  }
  if (experts % 8 == 0) {
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID6);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID6);
    copy_ubuf_to_gm(expert_counts->aligned+expert_counts->offset,totals,0,1,experts/8,0,0);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID6);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID6);
  }

  auto done=task_meta->aligned+task_meta->offset;
  done[1]=block_sum;
  *reinterpret_cast<volatile __gm__ int64_t *>(done)=epoch;
  direct_invalidate(done);
  pipe_barrier(PIPE_ALL);
}

extern "C" __aiv__ __attribute__((always_inline)) void _mlir_ciface_queue_acquire_vector(
    memref_t<__gm__ int64_t,1> *states, int64_t epoch, int32_t job,
    memref_t<__gm__ int64_t,1> *control) {
  pipe_barrier(PIPE_ALL);
  auto flag=states->aligned+states->offset+job*16;
  direct_await_epoch(flag, epoch);
  auto out=control->aligned+control->offset;
  out[0]=flag[5] ? -1 : flag[2];
  out[1]=flag[1];
  out[2]=flag[3];
  out[3]=flag[4];
  dcci(reinterpret_cast<__gm__ uint64_t *>(out), cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
  pipe_barrier(PIPE_ALL);
}

extern "C" __aiv__ __attribute__((always_inline)) void _mlir_ciface_queue_schedule(
    memref_t<__gm__ int64_t,1> *source_states, memref_t<__gm__ int64_t,1> *merged_states,
    memref_t<__gm__ int64_t,1> *task_meta, memref_t<__gm__ int64_t,1> *queue_states,
    int64_t epoch, int32_t world, int32_t experts, int32_t bm,
    memref_t<__ubuf__ int32_t,1> *scratch) {
  pipe_barrier(PIPE_ALL);
  auto meta=task_meta->aligned+task_meta->offset;
  direct_await_epoch(meta, epoch);
  int32_t jobs=meta[1];
  auto src=source_states->aligned+source_states->offset;
  auto merged=merged_states->aligned+merged_states->offset;
  auto queue=queue_states->aligned+queue_states->offset;
  auto active=scratch->aligned+scratch->offset;
  auto next=active+experts;
  int32_t count_active=0;
  for (int32_t e=0;e<experts;++e) {
    direct_invalidate(merged+e*16);
    if (merged[e*16+2]>0) active[count_active++]=e;
    next[e]=0;
  }
  // MTE2 destinations must be 32-byte aligned even when E is not a multiple of 4.
  auto snapshot=reinterpret_cast<__ubuf__ int64_t *>(active+((2*experts+7)/8)*8);
  int32_t emitted=0;
  while (emitted<jobs) {
    {
      set_flag(PIPE_S,PIPE_MTE2,EVENT_ID7);
      wait_flag(PIPE_S,PIPE_MTE2,EVENT_ID7);
      copy_gm_to_ubuf(snapshot,src,0,1,experts*world*4,0,0);
      set_flag(PIPE_MTE2,PIPE_S,EVENT_ID7);
      wait_flag(PIPE_MTE2,PIPE_S,EVENT_ID7);
    }
    for (int32_t a=0;a<count_active;++a) {
      int32_t e=active[a];
      int32_t start=next[e];
      int32_t total=merged[e*16+2];
      if (start>=total) continue;
      int32_t prefix=0;
      bool ready=true, failed=false;
      for (int32_t s=0;s<world;++s) {
        int32_t count=snapshot[(e*world+s)*16+2];
        int32_t end=start+bm-prefix<count ? start+bm-prefix : count;
        int32_t begin=start>prefix ? start-prefix : 0;
        bool tail=s==world-1 && start+bm>total;
        if (end>begin || tail) {
          uint64_t word=static_cast<uint64_t>(snapshot[(e*world+s)*16]);
          if ((word>>32)!=static_cast<uint32_t>(epoch)) ready=false;
          else if (word&0x40000000u) failed=true;
          else if ((word&0x3fffffffu)<static_cast<uint32_t>(end>0?end:0) || (tail && !(word&0x80000000u))) ready=false;
        }
        prefix+=count;
      }
      if (!ready && !failed) continue;
      auto q=queue+emitted*16;
      q[1]=merged[e*16+1];
      q[2]=total-start<bm ? total-start : bm;
      q[3]=e;
      q[4]=start;
      q[5]=failed;
      *reinterpret_cast<volatile __gm__ int64_t *>(q)=epoch;
      direct_invalidate(q);
      next[e]=start+bm;
      ++emitted;
    }
  }
  pipe_barrier(PIPE_ALL);
}

extern "C" __aiv__ __attribute__((always_inline)) void _mlir_ciface_stream_dispatch_output(
    memref_t<__gm__ int64_t,1> *requests,memref_t<__gm__ int64_t,1> *assignments,
    int64_t epoch,int32_t actors,int32_t owner,int32_t jobs,int32_t chunk,
    memref_t<__ubuf__ int32_t,1> *scratch) {
  pipe_barrier(PIPE_ALL);
  auto last=scratch->aligned+scratch->offset;
  for(int32_t a=0;a<actors;++a)last[a]=0;
  auto req=requests->aligned+requests->offset;
  auto out=assignments->aligned+assignments->offset;
  int32_t next=0,closed=0;
  while(closed<actors-1) {
    for(int32_t a=0;a<actors;++a) {
      if(a==owner || last[a]<0)continue;
      auto request=req+a*16;direct_invalidate(request);
      uint64_t ticket=*reinterpret_cast<volatile __gm__ uint64_t *>(request);
      if((ticket>>32)!=static_cast<uint32_t>(epoch))continue;
      int32_t sequence=static_cast<uint32_t>(ticket);
      if(sequence<=last[a])continue;
      auto dst=out+a*16;
      int32_t end=next+chunk<jobs ? next+chunk : jobs;
      dst[1]=next;dst[2]=end;
      if(next>=jobs){last[a]=-1;++closed;}else{last[a]=sequence;next=end;}
      *reinterpret_cast<volatile __gm__ uint64_t *>(dst)=ticket;
      direct_invalidate(dst);
    }
  }
  pipe_barrier(PIPE_ALL);
}

extern "C" __aiv__ __attribute__((always_inline)) void _mlir_ciface_stream_wait_assignment(
    memref_t<__gm__ int64_t,1> *assignment,int64_t ticket,
    memref_t<__gm__ int64_t,1> *control) {
  pipe_barrier(PIPE_ALL);
  auto src=assignment->aligned+assignment->offset;
  direct_await_epoch(src, ticket);
  auto dst=control->aligned+control->offset;
  dst[0]=src[1];dst[1]=src[2];direct_invalidate(dst);
  pipe_barrier(PIPE_ALL);
}

extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_moe_prepare_memory(
    memref_t<__gm__ bfloat16_t, 1> *input,
    memref_t<__gm__ bfloat16_t, 1> *source,
    memref_t<__gm__ float, 1> *slots, int32_t actor, int32_t actors, int32_t world,
    int64_t input_elements, int64_t slot_elements, int64_t slot_stride,
    memref_t<__ubuf__ int32_t, 1> *scratch) {
  auto arena = scratch->aligned + scratch->offset;
  const int32_t words = scratch->sizes[0] < 4096 ? scratch->sizes[0] : 4096;
  const int32_t halfwords = 2 * words;
  auto tile = reinterpret_cast<__ubuf__ uint16_t *>(arena);
  auto src = reinterpret_cast<__gm__ uint16_t *>(input->aligned + input->offset);
  auto dst = reinterpret_cast<__gm__ uint16_t *>(source->aligned + source->offset);
  pipe_barrier(PIPE_ALL);
  set_atomic_none();
  // Hidden-width padding makes every input and slot copy a whole DMA block.
  for (int64_t pos = static_cast<int64_t>(actor) * halfwords;
       pos < input_elements; pos += static_cast<int64_t>(actors) * halfwords) {
    const int64_t left = input_elements - pos;
    const int32_t count = left < halfwords ? left : halfwords;
    if ((reinterpret_cast<uint64_t>(src) & 31) == 0) {
      copy_gm_to_ubuf(tile, src + pos, 0, 1, count / 16, 0, 0);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    } else {
      for (int32_t i = 0; i < count; ++i) {
        tile[i] = src[pos + i];
      }
      set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    }
    copy_ubuf_to_gm(dst + pos, tile, 0, 1, count / 16, 0, 0);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
  }
  // The copy has drained. Fill the same address once and reuse it for zero stores.
  const int32_t per_source = actors / world;
  int64_t pos = static_cast<int64_t>(actor % per_source) * words;
  if (actor < world * per_source && pos < slot_elements) {
    auto zero = reinterpret_cast<__ubuf__ float *>(arena);
    set_mask_norm();
    set_vector_mask(0, static_cast<uint64_t>(-1));
    vector_dup(zero, 0.0f, words / 64, 1, 1, 8, 0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    auto slot = slots->aligned + slots->offset + (actor / per_source) * slot_stride;
    for (; pos < slot_elements; pos += static_cast<int64_t>(per_source) * words) {
      const int64_t left = slot_elements - pos;
      const int32_t count = left < words ? left : words;
      copy_ubuf_to_gm(slot + pos, zero, 0, 1, count / 8, 0, 0);
    }
  }
  pipe_barrier(PIPE_ALL);
}

// Form FP32 bits using scalar integer instructions. The DMA copies the bits as-is.
__aiv__ __attribute__((always_inline)) uint32_t
route_weight16_bits(uint16_t value, bool bf16) {
  if (bf16) {
    return static_cast<uint32_t>(value) << 16;
  }
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000) << 16;
  uint32_t exponent = (value >> 10) & 31;
  uint32_t fraction = value & 1023;
  if (exponent == 0) {
    if (fraction == 0) {
      return sign;
    }
    exponent = 113;
    while ((fraction & 1024) == 0) {
      fraction <<= 1;
      --exponent;
    }
  } else {
    exponent = exponent == 31 ? 255 : exponent + 112;
  }
  return sign | (exponent << 23) | ((fraction & 1023) << 13);
}

extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_moe_prepare_weights16(
    memref_t<__gm__ int16_t, 1> *probs,
    memref_t<__gm__ int32_t, 1> *experts,
    memref_t<__gm__ float, 1> *weights,
    int32_t actor, int32_t actors, int32_t world, int32_t tokens, int32_t capacity,
    int32_t padded_topk, int32_t rows, int64_t row_stride, int64_t slot_stride,
    int32_t bf16, memref_t<__ubuf__ int32_t, 1> *scratch) {
  auto input = probs->aligned + probs->offset;
  auto ids = experts->aligned + experts->offset;
  auto output = reinterpret_cast<__gm__ uint32_t *>(weights->aligned + weights->offset);
  auto tile = reinterpret_cast<__ubuf__ uint32_t *>(scratch->aligned + scratch->offset);
  const int32_t blocks = (tokens - 1) / rows + 1;
  // Match the routing worker partition so every read follows its own AIV's stores.
  pipe_barrier(PIPE_ALL);
  set_atomic_none();
  for (int64_t work = actor; work < static_cast<int64_t>(world) * blocks; work += actors) {
    const int32_t first = (work % blocks) * rows;
    const int32_t count = tokens - first < rows ? tokens - first : rows;
    const int64_t base = (work / blocks * capacity + first) * padded_topk;
    for (int32_t i = 0; i < count * padded_topk; ++i) {
      auto expert = ids + base + i;
      invalidate(expert);
      uint32_t weight = 0;
      if (*reinterpret_cast<volatile __gm__ int32_t *>(expert) >= 0) {
        auto prob = input + (first + i / padded_topk) * row_stride +
                    (i % padded_topk) * slot_stride;
        invalidate(prob);
        weight = route_weight16_bits(
            *reinterpret_cast<volatile __gm__ int16_t *>(prob), bf16);
      }
      tile[i] = weight;
    }
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    // Routing rows are padded to a multiple of eight FP32 elements.
    copy_ubuf_to_gm(output + base, tile, 0, 1, count * padded_topk / 8, 0, 0);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
  }
  pipe_barrier(PIPE_ALL);
}

extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_stream_advance_chunk(
    memref_t<__ubuf__ int32_t, 1> *routes,
    memref_t<__gm__ int64_t, 1> *progress, int32_t valid,
    int32_t source_routes, int32_t topk, int32_t block, int32_t chunks,
    int32_t lane, int32_t lanes, int32_t world) {
  pipe_barrier(PIPE_ALL);
  auto ids = routes->aligned + routes->offset;
  auto progress_base = progress->aligned + progress->offset;
  const int32_t block_routes = topk * block;
  const int32_t buckets = world * chunks;
  for (int32_t first = 0; first < buckets; first += 64) {
    uint64_t touched = 0;
    for (int32_t row = 0; row < valid; ++row) {
      const int32_t route = ids[row * routes->strides[0]];
      if (route < 0 || route >= source_routes * world) continue;
      const int32_t bucket =
          (route / source_routes) * chunks + (route % source_routes) / block_routes;
      const int32_t bit = bucket - first;
      if (bit >= 0 && bit < 64) {
        touched |= uint64_t{1} << bit;
      }
    }
    for (int32_t bit = 0; touched; ++bit, touched >>= 1) {
      if ((touched & 1) == 0) continue;
      const int32_t bucket = first + bit;
      const int32_t source = bucket / chunks;
      const int32_t lo = source * source_routes + (bucket % chunks) * block_routes;
      const int32_t end = (source + 1) * source_routes;
      const int32_t hi = lo + block_routes < end ? lo + block_routes : end;
      int32_t count = 0;
      for (int32_t row = 0; row < valid; ++row) {
        const int32_t route = ids[row * routes->strides[0]];
        count += route >= lo && route < hi;
      }
      auto flag = progress_base + (bucket * lanes + lane) * 16;
      direct_invalidate(flag);
      auto value = reinterpret_cast<volatile __gm__ int64_t *>(flag);
      *value = *value + count;
      direct_invalidate(flag);
    }
  }
  pipe_barrier(PIPE_ALL);
}

extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_stream_wait_chunk(
    memref_t<__gm__ int32_t, 1> *expected_count,
    memref_t<__gm__ int64_t, 1> *published,
    memref_t<__gm__ int64_t, 1> *progress,
    memref_t<__gm__ int64_t, 1> *errors,
    int64_t epoch, int32_t world, int32_t compute) {
  pipe_barrier(PIPE_ALL);
  direct_await_epoch(published->aligned + published->offset, epoch);
  auto count = expected_count->aligned + expected_count->offset;
  direct_invalidate(count);
  const int64_t expected = 2LL * *reinterpret_cast<volatile __gm__ int32_t *>(count);
  auto base = progress->aligned + progress->offset;
  auto faults = errors->aligned + errors->offset;
  while (true) {
    int64_t total = 0;
    for (int32_t lane = 0; lane < compute * 2; ++lane) {
      auto flag = base + lane * 16;
      direct_invalidate(flag);
      total += *reinterpret_cast<volatile __gm__ int64_t *>(flag);
    }
    if (total == expected) break;
    bool failed = false;
    for (int32_t i = 0; i < world + compute; ++i) {
      auto flag = faults + i * 16;
      direct_invalidate(flag);
      failed |= *reinterpret_cast<volatile __gm__ int64_t *>(flag) != 0;
    }
    if (failed) break;
  }
  pipe_barrier(PIPE_ALL);
}
