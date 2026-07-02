from backend_utils import VendorDescriptor  # noqa: E402

vendor_info = VendorDescriptor(
    vendor_name="nvidia",
    device_name="cuda",
    device_query_cmd="nvidia-smi",
    tle_enabled=True,
)
ARCH_MAP = {"9": "hopper", "8": "ampere"}


__all__ = ["*"]
