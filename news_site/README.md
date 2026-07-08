# News/Updates website — Tin ngành điện

Một trang web nhỏ hiển thị feed tin tức từ MongoDB (collection `dc_commodity.TinNganhDien`),
giao diện kiểu "updates" (giống kemvani.la/updates). **Độc lập** với dashboard chart và
`README_mongodb.md` — chỉ đọc thẳng collection tin tức.

## Cấu trúc

| File | Vai trò |
|---|---|
| `app.py` | Backend FastAPI: kết nối MongoDB + 3 endpoint (`/`, `/api/news`, `/api/tags`) |
| `index.html` | Trang feed (HTML/CSS/JS thuần, không cần build) |
| `run.ps1` | Script chạy nhanh |

Kết nối lấy từ `MONGO_URI` trong `.secrets.toml` ở thư mục gốc (hoặc biến môi trường `MONGO_URI`).

## Chạy

Từ thư mục gốc dự án (`q:\Coding\power\dashboard`):

```powershell
.\news_site\run.ps1
```

hoặc trực tiếp:

```powershell
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe -m uvicorn news_site.app:app --reload --port 8000
```

Rồi mở http://127.0.0.1:8000

## API

- `GET /api/news?type=all|news|document|press&q=<text>&tag=<mã>&important_only=true&skip=0&limit=30`
  — feed đã lọc, mới nhất trước, có phân trang.
- `GET /api/tags?limit=40` — các mã/từ khoá phổ biến (cho chip lọc).

## Tính năng UI

- Tab lọc theo loại: Tất cả / Tin tức / Công bố / Thông cáo.
- Ô tìm kiếm (tiêu đề + nội dung), chip lọc theo mã cổ phiếu / từ khoá.
- Toggle "Chỉ tin quan trọng" (dùng cờ `is_important` của feed news).
- Gom nhóm theo ngày, badge nguồn/loại/quan trọng, nút "Xem thêm" phân trang.
- Tự động sáng/tối theo hệ thống.

## Deploy sau này (tuỳ chọn)

Chỉ mình bạn xem thì chạy local là đủ. Nếu muốn public: đóng gói bằng Docker hoặc deploy lên
một dịch vụ chạy Python (Render/Railway/Fly.io), nhớ đặt `MONGO_URI` qua biến môi trường và
whitelist IP trong MongoDB Atlas.
