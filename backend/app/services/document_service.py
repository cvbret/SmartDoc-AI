from io import BytesIO

from fastapi import UploadFile
from pypdf import PdfReader


SUPPORTED_CONTENT_TYPES = {
    "text/plain",
    "application/pdf",
}


async def parse_document(file: UploadFile) -> str:
    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise ValueError(
            "暂不支持该文件类型"
        )

    content = await file.read() # 把上传文件的原始字节读入 Python 内存。

    if file.content_type == "text/plain":
        return _parse_txt(content)

    if file.content_type == "application/pdf":
        return _parse_pdf(content)

    raise ValueError(
        "无法识别文件类型"
    )


def _parse_txt(content: bytes) -> str:
    return content.decode(
        "utf-8"
    )


def _parse_pdf(content: bytes) -> str:
    reader = PdfReader(
        BytesIO(content) # 把字节流转换为文件对象
    )

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)