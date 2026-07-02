from backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="tsingmicro",
    device_name="txda",
    device_query_cmd="tsm_smi",
    dispatch_key="PrivateUse1",
    fp64_enabled=False,
    int64_enabled=False,
)

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]
