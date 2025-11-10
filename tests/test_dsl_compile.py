from perceptron import box, image, perceive, text
from perceptron import config as cfg


@perceive(max_tokens=32)
def describe_region(img):
    im = image(img)
    return im + text("What is in this box?") + box(1, 2, 3, 4, image=im)


def test_compile_task_no_execute():
    # Provide a tiny PNG header as bytes; width/height may be missing
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"0" * 10
    # Run in compile-only mode by clearing credentials
    with cfg(api_key=None, provider=None):
        res = describe_region(png_bytes)
    assert res.raw and isinstance(res.raw, dict)
    content = res.raw.get("content", [])
    # Should contain text and image entries
    kinds = [c.get("type") for c in content]
    assert "image" in kinds and "text" in kinds
