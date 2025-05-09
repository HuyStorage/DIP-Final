import cv2
import numpy as np

L = 256


def Erosion(imgin):
    if imgin.dtype == bool:
        imgin = imgin.astype(np.uint8) * 255  # True → 255, False → 0
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
    imgout = cv2.erode(imgin, w)
    return imgout


def Dilation(imgin):
    if imgin.dtype == bool:
        imgin = imgin.astype(np.uint8) * 255
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgout = cv2.dilate(imgin, w)
    return imgout


def BoundaryExtraction(imgin):
    if imgin.dtype == bool:
        imgin = imgin.astype(np.uint8) * 255
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    temp = cv2.erode(imgin, w)
    imgout = imgin - temp
    return imgout


def Contour(imgin):
    M, N = imgin.shape
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    n = len(contour)
    for i in range(n):
        x1 = contour[i][0][0]
        y1 = contour[i][0][1]

        x2 = contour[(i + 1)%n][0][0]
        y2 = contour[(i + 1)%n][0][1]

        cv2.line(imgout, (x1, y1), (x2, y2), (0, 255, 0), 2)

    x1 = contour[n - 1][0][0]
    y1 = contour[n - 1][0][1]

    x2 = contour[0][0][0]
    y2 = contour[0][0][1]

    cv2.line(imgout, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return imgout

def ConnectedComponents(imgin):
    nguong = 200
    _, temp = cv2.threshold(imgin, nguong, (L - 1), cv2.THRESH_BINARY)
    # Xóa nhiễu lốm đốm (nhiễu xung)
    imgout = cv2.medianBlur(temp, 7)
    n, label = cv2.connectedComponents(imgout)

    a = np.zeros(n, np.int32)

    # label là ảnh mà điểm ảnh là số nguyên 32bit
    # nghĩa là label là mảng 2 chiều mà mỗi phần tử là số nguyên 32bit
    # Ảnh của ta không thể thể có đến 2 tỷ đối tượng nên quá dư để dùng
    M, N = label.shape

    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            # r = 0 là màu nền, ta không cần xét
            if r > 0:
                a[r] = a[r] + 1
    s = "Co %d thanh phan lien thong" % (
        n - 1
    )  # -1 vì background cũng là thành phần liên thông
    cv2.putText(imgout, s, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    for r in range(1, n):
        s = "%3d %5d" % (r, a[r])
        cv2.putText(
            imgout,
            s,
            (10, 20 * (r + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
        )

    return imgout


def RemoveSmallRice(imgin):
    # 81 là kích thước lớn nhất của hạt gạo
    # Làm đậm bóng đổ bnafwg biến đổi top-hat
    w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
    temp = cv2.morphologyEx(imgin, cv2.MORPH_TOPHAT, w)

    # Phân ngưỡng để có được ảnh nhị phân (ảnh trắng đen)
    nguong = 100
    _, temp = cv2.threshold(temp, nguong, L - 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    n, label = cv2.connectedComponents(temp)
    a = np.zeros(n, np.int32)

    M, N = label.shape

    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            # r = 0 là màu nền, ta không cần xét
            if r > 0:
                a[r] = a[r] + 1

    # Xóa hạt gạo nhỏ hơn 0.7 * max_value
    max_value = np.max(a)

    imgout = np.zeros((M, N), np.uint8)
    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            if r > 0:
                if a[r] >= 0.7 * max_value:
                    imgout[x, y] = L - 1
    return imgout