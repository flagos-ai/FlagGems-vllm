from backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="aipu",
    device_name="aipu",
    device_query_cmd="aipu",
    dispatch_key="PrivateUse1",
    fp64_enabled=False,
    bf16_enabled=False,
    int64_enabled=False,
)

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]
