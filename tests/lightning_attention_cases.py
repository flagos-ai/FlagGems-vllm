# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrefixCase:
    group: str
    b: int
    h: int
    t: int
    d: int
    e: int
    layout: str = "contiguous"
    initial: str = "random"
    state_layout: str = "contiguous"
    dtype: torch.dtype = torch.float32
    decay: str = "heads"
    signal: str = "random"


@dataclass(frozen=True)
class DecodeCase:
    group: str
    b: int
    h: int
    d: int
    slots: str = "random"
    layout: str = "contiguous"
    state_layout: str = "contiguous"
    dtype: torch.dtype = torch.float32
    block: int = 32
    decay: str = "heads"
    metadata_strided: bool = False
    signal: str = "random"


def _prefix(group, rows):
    return [PrefixCase(group, *row) for row in rows]


def _decode(group, rows):
    return [DecodeCase(group, *row) for row in rows]


_BF = torch.bfloat16
_FP = torch.float32

PREFIX_CASES = [
    *_prefix(
        "tiny",
        [
            (1, 2, 0, 64, 64),
            (1, 3, 0, 128, 128, "contiguous", "none"),
            (2, 3, 1, 64, 64),
            (1, 8, 2, 128, 128),
            (3, 1, 3, 64, 32),
            (2, 3, 4, 128, 128, "stride2", "none"),
            (1, 4, 5, 64, 64),
            (2, 2, 7, 128, 64),
            (1, 8, 8, 64, 64),
            (1, 16, 16, 128, 128),
        ],
    ),
    *_prefix(
        "tile",
        [
            (1, 4, 31, 64, 64),
            (2, 2, 32, 128, 128),
            (1, 8, 33, 64, 64),
            (1, 3, 62, 128, 64),
            (1, 2, 63, 64, 64, "stride2", "none"),
            (1, 8, 64, 64, 64),
            (1, 2, 64, 128, 64),
            (1, 2, 65, 64, 64, "contiguous", "random", "contiguous", _BF),
            (1, 4, 127, 64, 64),
            (1, 8, 129, 128, 128),
        ],
    ),
    *_prefix(
        "history",
        [
            (2, 2, 192, 64, 64),
            (1, 4, 254, 128, 128),
            (2, 3, 255, 64, 64, "stride2"),
            (1, 2, 256, 128, 128, "contiguous", "none"),
            (1, 8, 257, 64, 64),
            (1, 2, 257, 64, 64, "stride2"),
            (1, 4, 511, 64, 64),
            (1, 16, 512, 128, 128),
            (1, 2, 513, 128, 128, "contiguous", "random", "contiguous", _BF),
            (1, 3, 769, 64, 64),
        ],
    ),
    *_prefix(
        "scan",
        [
            (1, 16, 1024, 128, 128),
            (1, 4, 1408, 64, 64),
            (1, 4, 1471, 128, 128),
            (1, 4, 1472, 128, 128),
            (1, 4, 1473, 128, 128),
            (1, 8, 1535, 64, 64),
            (1, 8, 1536, 64, 64),
            (1, 8, 1537, 64, 64),
            (1, 4, 2047, 128, 128),
            (1, 16, 2048, 128, 128),
        ],
    ),
    *_prefix(
        "long",
        [
            (1, 4, 2049, 128, 128),
            (2, 2, 3072, 64, 64),
            (1, 4, 4095, 64, 64),
            (1, 8, 4096, 128, 128),
            (1, 4, 4097, 128, 128),
            (1, 2, 8191, 64, 64),
            (1, 8, 8192, 128, 128),
            (1, 2, 12288, 128, 128),
            (1, 2, 16384, 64, 64),
            (1, 2, 32768, 128, 128),
        ],
    ),
    *_prefix(
        "serving",
        [
            (4, 1, 96, 64, 64),
            (1, 32, 128, 128, 128),
            (2, 8, 320, 64, 64),
            (2, 2, 320, 64, 32),
            (3, 3, 384, 128, 128),
            (4, 8, 512, 64, 64),
            (8, 2, 160, 128, 128),
            (2, 16, 768, 64, 64),
            (1, 6, 1025, 128, 128),
            (1, 32, 2048, 64, 64),
        ],
    ),
    *_prefix(
        "layout",
        [
            (1, 2, 3, 64, 1, "offset"),
            (1, 3, 19, 64, 7, "packed"),
            (2, 2, 67, 128, 17, "token-major", "random", "pitched"),
            (1, 4, 128, 64, 63, "stride2", "random", "interleaved"),
            (1, 5, 195, 64, 65, "mixed"),
            (2, 2, 258, 128, 96, "packed", "random", "transposed"),
            (2, 3, 319, 64, 127, "offset"),
            (1, 2, 513, 128, 129, "stride2"),
            (1, 4, 1553, 128, 32, "mixed", "random", "pitched"),
            (2, 2, 2179, 64, 64, "packed", "random", "head-major"),
        ],
    ),
    *_prefix(
        "numerics",
        [
            (1, 2, 11, 64, 64, "contiguous", "zero", "contiguous", _BF, "zero"),
            (2, 3, 69, 128, 128, "packed", "none", "contiguous", _BF, "tiny"),
            (1, 3, 255, 128, 64, "contiguous", "random", "pitched", _BF, "steep"),
            (
                1,
                4,
                256,
                64,
                64,
                "contiguous",
                "random",
                "contiguous",
                _FP,
                "steep",
                "zero-q",
            ),
            (
                1,
                4,
                257,
                64,
                64,
                "contiguous",
                "random",
                "contiguous",
                _FP,
                "half",
                "zero-k",
            ),
            (
                2,
                2,
                510,
                128,
                128,
                "mixed",
                "random",
                "contiguous",
                _FP,
                "half",
                "zero-v",
            ),
            (
                1,
                3,
                515,
                64,
                64,
                "contiguous",
                "zero",
                "contiguous",
                _BF,
                "mixed",
                "alternating",
            ),
            (1, 2, 1474, 128, 128, "token-major", "none", "contiguous", _BF, "steep"),
            (1, 4, 2050, 64, 64, "packed", "random", "contiguous", _FP, "zero"),
            (1, 2, 4098, 128, 128, "contiguous", "random", "contiguous", _BF, "tiny"),
        ],
    ),
]

DECODE_CASES = [
    *_decode(
        "small",
        [
            (1, 1, 64),
            (1, 2, 128),
            (1, 8, 64),
            (1, 8, 128),
            (1, 16, 128),
            (2, 3, 64),
            (2, 8, 128),
            (3, 4, 64),
            (5, 4, 128, "half-pad"),
            (7, 8, 64),
        ],
    ),
    *_decode(
        "dispatch",
        [
            (7, 9, 128),
            (8, 8, 128),
            (5, 13, 128),
            (1, 63, 128),
            (1, 64, 128),
            (1, 65, 128),
            (15, 4, 64),
            (16, 4, 64),
            (17, 4, 64),
            (9, 8, 128),
        ],
    ),
    *_decode(
        "throughput",
        [
            (8, 8, 64),
            (16, 8, 128),
            (24, 16, 64),
            (32, 16, 64),
            (32, 16, 128),
            (48, 16, 128),
            (64, 16, 128),
            (96, 8, 64),
            (128, 8, 128),
            (256, 4, 64),
        ],
    ),
    *_decode(
        "slots",
        [
            (3, 5, 64, "linear"),
            (4, 4, 128, "reverse"),
            (5, 4, 64, "sparse"),
            (6, 7, 128, "random"),
            (8, 3, 64, "reverse"),
            (13, 8, 128, "sparse"),
            (19, 4, 64, "linear"),
            (31, 8, 128, "reverse"),
            (40, 8, 64, "sparse"),
            (64, 8, 128, "sparse"),
        ],
    ),
    *_decode(
        "padding",
        [
            (1, 4, 64, "all-pad"),
            (2, 8, 128, "one-live"),
            (5, 4, 64, "half-pad"),
            (7, 3, 128, "tail-pad"),
            (8, 8, 64, "all-pad"),
            (15, 4, 128, "half-pad"),
            (17, 2, 64, "one-live"),
            (33, 8, 128, "tail-pad"),
            (64, 4, 64, "half-pad"),
            (96, 4, 128, "one-live"),
        ],
    ),
    *_decode(
        "layout",
        [
            (1, 4, 64, "linear", "offset"),
            (3, 8, 128, "reverse", "stride2"),
            (4, 3, 64, "random", "packed"),
            (5, 4, 128, "sparse", "mixed", "pitched"),
            (8, 4, 64, "half-pad", "contiguous", "interleaved"),
            (9, 8, 128, "random", "contiguous", "transposed"),
            (16, 4, 64, "reverse", "offset", "pitched"),
            (24, 8, 128, "random", "packed", "head-major"),
            (32, 8, 64, "linear", "contiguous", "transposed"),
            (48, 4, 128, "half-pad", "mixed", "interleaved"),
        ],
    ),
    *_decode(
        "numerics",
        [
            (1, 3, 64, "linear", "contiguous", "contiguous", _BF, 32, "zero"),
            (3, 5, 128, "random", "contiguous", "contiguous", _BF, 32, "tiny"),
            (5, 4, 64, "half-pad", "contiguous", "contiguous", _BF, 32, "heads"),
            (7, 2, 128, "reverse", "packed", "pitched", _BF, 32, "steep"),
            (
                8,
                3,
                64,
                "sparse",
                "contiguous",
                "contiguous",
                _FP,
                32,
                "half",
                False,
                "zero-q",
            ),
            (
                12,
                5,
                128,
                "random",
                "contiguous",
                "contiguous",
                _FP,
                32,
                "mixed",
                False,
                "zero-k",
            ),
            (
                16,
                8,
                64,
                "tail-pad",
                "contiguous",
                "contiguous",
                _BF,
                32,
                "zero",
                False,
                "zero-v",
            ),
            (
                23,
                4,
                128,
                "linear",
                "contiguous",
                "contiguous",
                _BF,
                32,
                "steep",
                False,
                "alternating",
            ),
            (32, 4, 64, "half-pad", "stride2", "interleaved", _BF, 32, "mixed"),
            (64, 4, 128, "random", "contiguous", "contiguous", _BF, 32, "tiny"),
        ],
    ),
    *_decode(
        "metadata",
        [
            (2, 4, 64, "linear", "contiguous", "contiguous", _FP, 8),
            (4, 4, 128, "random", "contiguous", "contiguous", _FP, 16),
            (5, 6, 64, "sparse", "contiguous", "contiguous", _FP, 64),
            (8, 4, 128, "reverse", "contiguous", "contiguous", _FP, 128),
            (9, 3, 64, "half-pad", "contiguous", "pitched", _FP, 16, "heads", True),
            (12, 4, 128, "sparse", "contiguous", "contiguous", _FP, 32, "mixed", True),
            (18, 4, 64, "tail-pad", "packed", "contiguous", _FP, 8, "tiny", True),
            (24, 6, 128, "linear", "contiguous", "head-major", _BF, 16, "half", True),
            (33, 4, 64, "random", "contiguous", "contiguous", _FP, 32, "zero", True),
            (48, 8, 128, "reverse", "contiguous", "contiguous", _FP, 64, "heads", True),
        ],
    ),
]


def case_id(case):
    tag = "p" if isinstance(case, PrefixCase) else "d"
    t = getattr(case, "t", 1)
    e = getattr(case, "e", case.d)
    return f"{tag}-{case.group}-b{case.b}h{case.h}t{t}d{case.d}e{e}"


def make_qkv(case, device):
    b, h, t, d, e = (
        case.b,
        case.h,
        getattr(case, "t", 1),
        case.d,
        getattr(case, "e", case.d),
    )

    def random(*shape):
        return torch.randn(*shape, device=device, dtype=case.dtype).mul_(0.125)

    if case.layout == "packed":
        raw = random(b, t, h * (2 * d + e))
        qkv = tuple(
            x.view(b, t, h, n).transpose(1, 2)
            for x, n in zip(raw.split([h * d, h * d, h * e], -1), (d, d, e))
        )
    else:

        def tensor(width, layout):
            if layout == "token-major":
                return random(b, t, h, width).transpose(1, 2)
            if layout == "offset":
                return random(b, h, t, width + 3)[..., 1 : 1 + width]
            if layout == "stride2":
                return random(b, t, h, width * 2)[..., ::2].transpose(1, 2)
            return random(b, h, t, width)

        layouts = (
            ("stride2", "contiguous", "token-major")
            if case.layout == "mixed"
            else (case.layout,) * 3
        )
        qkv = tuple(tensor(width, layout) for width, layout in zip((d, d, e), layouts))
    if case.signal.startswith("zero-"):
        qkv[{"zero-q": 0, "zero-k": 1, "zero-v": 2}[case.signal]].zero_()
    elif case.signal == "alternating":
        for x in qkv:
            x[..., 1::2].neg_()
        qkv[1][..., ::2].copy_(qkv[1][..., 1::2])
        qkv[0][..., 1::2].copy_(-0.9 * qkv[0][..., ::2])
    return qkv


def make_slope(case, device):
    h = case.h
    if case.decay == "heads":
        return torch.linspace(0.002, 0.12, h, device=device)
    if case.decay == "steep":
        return torch.linspace(1.0, 8.0, h, device=device)
    if case.decay == "mixed":
        rates = torch.tensor([0.0, 1e-6, 0.125, 3.0], device=device)
        return rates[torch.arange(h, device=device) % 4]
    return torch.full(
        (h,),
        {"zero": 0.0, "tiny": 1e-6, "half": 0.6931471805599453}[case.decay],
        device=device,
    )


def make_state(case, rows, device):
    d, e, h = case.d, getattr(case, "e", case.d), case.h
    layout = case.state_layout
    if layout == "head-major":
        return torch.empty(h, rows, d, e, device=device).transpose(0, 1)
    if layout == "transposed":
        return torch.empty(rows, h, e, d, device=device).transpose(-1, -2)
    if layout == "interleaved":
        return torch.empty(rows, h, d, 2 * e, device=device)[..., ::2]
    if layout == "pitched":
        return torch.empty(rows, h, d, e + 7, device=device)[..., 2 : 2 + e]
    return torch.empty(rows, h, d, e, device=device)


def make_initial(case, device):
    if case.initial == "none":
        return None
    state = make_state(case, case.b, device)
    return state.zero_() if case.initial == "zero" else state.normal_(std=0.125)


def make_decode_inputs(case, device):
    b = case.b
    rows = 2 * b + 3 if case.slots == "sparse" else b + 3
    seq = torch.arange(b, device=device)
    if case.slots == "linear":
        slots = seq
    elif case.slots == "reverse":
        slots = seq.flip(0)
    elif case.slots == "sparse":
        slots = 2 * seq + 1
    else:
        slots = torch.randperm(rows, device=device)[:b]
        if case.slots == "half-pad":
            slots[1::2] = -1
        elif case.slots == "tail-pad":
            slots[(b + 1) // 2 :] = -1
        elif case.slots == "one-live":
            slots[:-1] = -1
        elif case.slots == "all-pad":
            slots.fill_(-1)
    slots = slots.to(torch.int64 if b % 2 == 0 else torch.int32)
    slope = make_slope(case, device)
    if case.metadata_strided:
        slot_storage = torch.full((b, 2), -1, dtype=slots.dtype, device=device)
        slot_storage[:, 0] = slots
        slots = slot_storage[:, 0]
        rate_storage = torch.zeros(case.h, 2, device=device)
        rate_storage[:, 0] = slope
        slope = rate_storage[:, 0]
    return slope, slots, make_state(case, rows, device).normal_(std=0.125)


def native_prefix_reason(case):
    if case.t == 0:
        return "empty sequence"
    if case.e < 16 or case.e & (case.e - 1):
        return "native value tile needs a power of two >=16"
    if case.dtype == torch.bfloat16:
        return "native prefix needs FP32 V with FP32 decay"
    return None


def native_decode_reason(qkv, slope, slots):
    if not all(x.is_contiguous() for x in qkv):
        return "native decode assumes shared QKV strides and unit feature stride"
    if slope.stride(0) != 1 or slots.stride(0) != 1:
        return "native decode ignores slope/slot strides"
    return None
