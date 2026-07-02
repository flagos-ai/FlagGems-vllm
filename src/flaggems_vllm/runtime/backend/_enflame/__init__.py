from backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="enflame",
    device_name="gcu",
    device_query_cmd="",
    dispatch_key="PrivateUse1",
    fp64_enabled=False,
    int64_enabled=False,
    tle_enabled=True,
)

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]
