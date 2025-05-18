import numpy as np
import argparse
import cv2

def embed_parity_watermark(binary_img, watermark_bits, block_size=(8, 8)):
    """
    在二值图像中嵌入块奇偶校验水印
    :param binary_img: 输入二值图像，numpy数组，像素值为0或1
    :param watermark_bits: 待嵌入的水印比特列表或字符串，如'010101'
    :param block_size: 块大小，默认8x8
    :return: 嵌入水印后的图像
    """
    img = binary_img.copy()
    h, w = img.shape
    bh, bw = block_size
    wm_idx = 0
    wm_len = len(watermark_bits)
    for i in range(0, h, bh):
        for j in range(0, w, bw):
            if wm_idx >= wm_len:
                break
            block = img[i:i+bh, j:j+bw]
            block_sum = np.sum(block)
            bit = int(watermark_bits[wm_idx])
            if block_sum % 2 != bit:
                # 奇偶性不符，随机翻转一个像素
                flat_idx = np.random.choice(block.size)
                x, y = np.unravel_index(flat_idx, block.shape)
                block[x, y] = 1 - block[x, y]
                img[i:i+bh, j:j+bw] = block
            wm_idx += 1
    return img

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="块奇偶校验水印嵌入")
    parser.add_argument('--input', type=str, required=True, help='输入二值图像路径')
    parser.add_argument('--output', type=str, required=True, help='输出水印图像路径')
    parser.add_argument('--watermark', type=str, required=True, help='水印比特串，如101010')
    parser.add_argument('--blocksize', type=int, nargs=2, default=[8, 8], help='块大小，如 8 8')

    args = parser.parse_args()

    # 读取并二值化图像
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

    # 嵌入水印
    watermarked_img = embed_parity_watermark(binary_img, args.watermark, tuple(args.blocksize))

    # 保存结果
    cv2.imwrite(args.output, watermarked_img * 255)
    print(f"水印嵌入完成，结果已保存到 {args.output}")
