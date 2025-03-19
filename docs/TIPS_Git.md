# Các lệnh cơ bản

- Push các file sửa đổi và thêm mới lên server:
```python
    # lan dau, phai khoi tao
    git config --global user.name 'lenhoanh'
    git config --global user.email 'lenhoanh@gmail.com'

    # đưa tất cả các files từ Thư mục đang làm việc sang Staging Area
    git add .  

    # hoặc đưa 1 file
    git add filename 

    # đưa trở lại các file từ Staging Area về lại Thư mục làm việc
    git restore --staged filename

    # đưa các file từ Staging Area sang Committed
    git commit -m "Add new files"

    # push các file lên server
    git push

    # bat buoc push len
    git push --force
```

- hiển thị lịch sử các lệnh commit: 
```python
    git log
    git reflog
        a9107e7 HEAD@{7}: reset: moving to HEAD
        a9107e7 HEAD@{8}: commit: Upload GMM_DAE, libs/yolov5
        df7f9f3 HEAD@{9}: reset: moving to HEAD
        df7f9f3 HEAD@{10}: commit: Upload GMM_DAE
    # quay tro lai diem commit a9107e7
    git reset --hard a9107e7
```

- Show các file đã sửa đổi nhưng chưa được chuyển sang Staging Area
```python
    git diff --name-only
```

- Hiển thị các nhánh
```python
    # Hiển thị các nhánh
    git branch

    # Tạo nhánh mới
    git checkout -b new_branch_name

    # Chuyển sang nhánh khác (có tên là branch_name)
    git checkout branch_name

    # Đưa commit từ branch phụ (branch_name) vào branch main
    git merge branch_name

    # pull code from server to local
    git pull
```
