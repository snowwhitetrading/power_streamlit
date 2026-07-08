# Đưa website lên internet (Render — miễn phí)

Sau khi làm xong, bạn có một URL kiểu `https://tin-nganh-dien.onrender.com` chạy 24/7,
không cần bật máy hay chạy local. App đọc dữ liệu thẳng từ MongoDB Atlas (đã có sẵn).

> Trang để **mở** (không mật khẩu) theo lựa chọn hiện tại. Muốn thêm đăng nhập sau, báo lại.

## Chuẩn bị đã xong (trong repo)
- `news_site/requirements.txt` — thư viện cần cài
- `render.yaml` — cấu hình deploy tự động
- `.gitignore` — đã loại `.secrets.toml`, `data/`, `.venv` (secrets KHÔNG lên git)

## Các bước (một lần, ~10 phút)

### 1. Đẩy code lên GitHub
- Tạo tài khoản https://github.com (nếu chưa có) và một repo mới (Private cũng được).
- Trong thư mục dự án:
  ```bash
  git add news_site render.yaml .gitignore
  git commit -m "Add news website + Render deploy config"
  git remote add origin https://github.com/<tên-bạn>/<tên-repo>.git
  git push -u origin main
  ```
  (Kiểm tra `git status` chắc chắn KHÔNG có `.secrets.toml` trong danh sách được commit.)

### 2. Tạo service trên Render
- Đăng ký https://render.com (đăng nhập bằng GitHub cho nhanh).
- **New +** → **Blueprint** → chọn repo vừa đẩy → Render tự đọc `render.yaml`.
- Khi hỏi biến môi trường **`MONGO_URI`**, dán chuỗi kết nối Atlas của bạn
  (lấy trong `.secrets.toml`, dòng `MONGO_URI = "mongodb+srv://..."`).
- Bấm **Apply/Create** → chờ build ~2–3 phút.

### 3. Cho Render kết nối MongoDB Atlas
- Vào MongoDB Atlas → **Network Access** → **Add IP Address** → **Allow access from anywhere**
  (`0.0.0.0/0`). (Render free dùng IP động nên cần mở; dữ liệu vẫn cần đúng user/mật khẩu để đọc.)

Xong. Mở URL Render cấp là thấy trang.

## Lưu ý
- **Gói free ngủ khi không dùng:** sau ~15 phút không ai vào, lần mở kế tiếp chờ ~50 giây rồi
  chạy bình thường. Muốn luôn tức thì thì nâng gói trả phí (~7 USD/tháng) — không bắt buộc.
- **Cập nhật code sau này:** chỉ cần `git push`, Render tự build lại.
- **Muốn tên miền riêng** (vd `tin.tencuaban.com`): Render → Settings → Custom Domain, miễn phí.
- Pipeline cập nhật tin (scheduled task trên máy bạn) **giữ nguyên** — nó ghi vào cùng MongoDB,
  website trên Render đọc ra, nên tin vẫn tự cập nhật mà không liên quan tới việc deploy.
