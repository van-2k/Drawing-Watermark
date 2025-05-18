import numpy as np
import argparse
import cv2

def extract_parity_watermark(watermarked_img, num_bits, block_size=(8, 8)):
    h, w = watermarked_img.shape
    bh, bw = block_size
    bits = []
    for i in range(0, h, bh):
        for j in range(0, w, bw):
            if len(bits) >= num_bits:
                break
            block = watermarked_img[i:i+bh, j:j+bw]
            block_sum = np.sum(block)
            bits.append(str(block_sum % 2))
    return ''.join(bits)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="块奇偶校验水印提取")
    parser.add_argument('--input', type=str, required=True, help='输入水印图像路径')
    parser.add_argument('--num_bits', type=int, required=True, help='要提取的水印比特数')
    parser.add_argument('--blocksize', type=int, nargs=2, default=[8, 8], help='块大小，如 8 8')

    args = parser.parse_args()

    # 读取并二值化图像
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

    # 提取水印
    watermark = extract_parity_watermark(binary_img, args.num_bits, tuple(args.blocksize))

    print(f"提取的水印比特串为: {watermark}")
