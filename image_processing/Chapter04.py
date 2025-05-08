import numpy as np
import cv2

L = 256


def Spectrum(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)

    # Bước 1 và 2:
    # Tạo ảnh mới có kích thước PxQ
    # và thêm số 0 và phần mở rộng
    fp = np.zeros((P, Q), np.float32)
    fp[:M, :N] = imgin
    fp = fp / (L - 1)

    # Bước 3:
    # Nhân (-1)^(x+y) để dời vào tâm ảnh
    for x in range(0, M):
        for y in range(0, N):
            if (x + y) % 2 == 1:
                fp[x, y] = -fp[x, y]

    # Bước 4:
    # Tính DFT
    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)

    # Tính spectrum
    S = np.sqrt(F[:, :, 0] ** 2 + F[:, :, 1] ** 2)
    S = np.clip(S, 0, L - 1)
    S = S.astype(np.uint8)
    return S


def FrequencyFiltering(imgin, H):
    f = imgin.astype(np.float32)
    # Buoc 1: DFT
    F = np.fft.fft2(f)
    # Buoc 2: Shift vao center of the image
    F = np.fft.fftshift(F)

    # Buoc 3: Nhan F voi H
    G = F * H

    # Buoc 4: Shift ra tro lai
    G = np.fft.ifftshift(G)

    # Buoc 5: IDFT
    g = np.fft.ifft2(G)
    gR = np.clip(g.real, 0, L - 1)
    imgout = gR.astype(np.uint8)
    return imgout


def CreateMoireFilter(M, N):
    H = np.ones((M, N), np.complex64)
    H.imag = 0.0
    u1 = 44
    v1 = 55

    u2 = 85
    v2 = 55

    u3 = 41
    v3 = 111

    u4 = 81
    v4 = 111

    u5 = M - u1
    v5 = N - v1

    u6 = M - u2
    v6 = N - v2

    u7 = M - u3
    v7 = N - v3

    u8 = M - u4
    v8 = N - v4

    D0 = 10

    for u in range(0, M):
        for v in range(0, N):
            Duv = np.sqrt((1.0 * u - u1) ** 2 + (1.0 * v - v1) ** 2)
            if Duv <= D0:
                H.real[u, v] = 0.0

            Duv = np.sqrt((1.0 * u - u2) ** 2 + (1.0 * v - v2) ** 2)
            if Duv <= D0:
                H.real[u, v] = 0.0

            Duv = np.sqrt((1.0 * u - u3) ** 2 + (1.0 * v - v3) ** 2)
            if Duv <= D0:
                H.real[u, v] = 0.0

            Duv = np.sqrt((1.0 * u - u4) ** 2 + (1.0 * v - v4) ** 2)
            if Duv <= D0:
                H.real[u, v] = 0.0

            Duv = np.sqrt((1.0 * u - u5) ** 2 + (1.0 * v - v5) ** 2)
            if Duv <= D0:
                H.real[u, v] = 0.0

            Duv = np.sqrt((1.0 * u - u6) ** 2 + (1.0 * v - v6) ** 2)
            if Duv <= D0:
                H.real[u, v] = 0.0

            Duv = np.sqrt((1.0 * u - u7) ** 2 + (1.0 * v - v7) ** 2)
            if Duv <= D0:
                H.real[u, v] = 0.0

            Duv = np.sqrt((1.0 * u - u8) ** 2 + (1.0 * v - v8) ** 2)
            if Duv <= D0:
                H.real[u, v] = 0.0

    return H


def FreFilter(imgin):
    M, N = imgin.shape

    # B1
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)

    fp = np.zeros((P, Q), np.float32)

    # B2
    fp[:M, :N] = 1.0 * imgin

    #   B3
    for x in range(0, M):
        for y in range(0, N):
            if (x + y) % 2 == 1:
                fp[x, y] = -fp[x, y]

    # B4
    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)

    # B5:
    H = CreateMoireFilter(P, Q)

    # B6:
    G = cv2.mulSpectrums(F, H, flags=cv2.DFT_ROWS)

    # B7:
    g = cv2.idft(G, flags=cv2.DFT_SCALE)

    # B8:
    gR = g[:M, :N, 0]
    for x in range(0, M):
        for y in range(0, N):
            if (x + y) % 2 == 1:
                gR[x, y] = -gR[x, y]

    gR = np.clip(gR, 0, L - 1)
    imgout = gR.astype(np.uint8)

    return imgout


def CreateInterInferenceFilter(M, N):
    H = np.ones((M, N), np.complex64)
    H.imag = 0.0
    D0 = 7
    D1 = 7
    for u in range(0, M):
        for v in range(0, N):
            if u not in range(M // 2 - D0, N // 2 + D0 + 1):
                if abs(v - N // 2) <= D1:
                    H.real[u, v] = 0.0

    return H


def RemoveMoire(imgin):
    M, N = imgin.shape
    H = CreateMoireFilter(M, N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout


def RemoveInterInference(imgin):
    M, N = imgin.shape
    H = CreateInterInferenceFilter(M, N)
    imgout = FrequencyFiltering(imgin, H)

    return imgout


def CreateMotionFilter(M, N):
    H = np.zeros((M, N), np.complex64)
    T = 1.0
    a = 0.1
    b = 0.1
    phi_prev = 0.0
    for u in range(0, M):
        for v in range(0, N):
            phi = np.pi * ((u - M // 2) * a + (v - N // 2) * b)
            if abs(phi) < 1.0e-6:
                phi = phi_prev
            RE = T * np.sin(phi) * np.cos(phi) / phi
            IM = -T * np.sin(phi) * np.sin(phi) / phi
            H.real[u, v] = RE
            H.imag[u, v] = IM
            phi_prev = phi
    return H


def CreateMotion(imgin):
    M, N = imgin.shape
    H = CreateMotionFilter(M, N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout


def CreateDemotionFilter(M, N):
    H = np.zeros((M, N), np.complex64)
    T = 1.0
    a = 0.1
    b = 0.1
    phi_prev = 0.0
    for u in range(0, M):
        for v in range(0, N):
            phi = np.pi * ((u - M // 2) * a + (v - N // 2) * b)
            mau_so = np.sin(phi)

            if abs(mau_so) < 1.0e-6:
                phi = phi_prev
            
            RE = phi / (T * np.sin(phi)) * np.cos(phi)
            IM = phi / T
            H.real[u, v] = RE
            H.imag[u, v] = IM
            phi_prev = phi
    return H


def Demotion(imgin):
    M, N = imgin.shape
    H = CreateDemotionFilter(M, N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout


def DemotionNoise(imgin):
    temp = cv2.medianBlur(imgin, 7)
    return Demotion(temp)