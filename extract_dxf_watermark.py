import ezdxf
import numpy as np
import argparse

def is_horizontal(line, angle_tol=5):
    start = np.array(line.dxf.start)
    end = np.array(line.dxf.end)
    vec = end - start
    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    return abs(angle) < angle_tol or abs(angle - 180) < angle_tol

def angle_between(a, b, c):
    # 计算∠abc的度数
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    return np.degrees(angle)

def point_on_segment(p, a, b, tol=1.0):
    # 判断点p是否在ab线段上（允许一定容差）
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    if t < 0 or t > 1:
        return False
    closest = a + t * ab
    return np.linalg.norm(closest - p) < tol

def extract_watermark_dxf(input_path, num_bits):
    doc = ezdxf.readfile(input_path)
    msp = doc.modelspace()

    # 1. 获取主线（只考虑靠近y_min的最长水平主线）
    xs = []
    ys = []
    for e in msp:
        if e.dxftype() == 'LINE':
            xs.extend([e.dxf.start[0], e.dxf.end[0]])
            ys.extend([e.dxf.start[1], e.dxf.end[1]])
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    main_line = None
    max_length = 0
    y_tol = 1.0  # 容差，与嵌入脚本一致
    for e in msp:
        if e.dxftype() == 'LINE' and is_horizontal(e):
            start = np.array(e.dxf.start)
            end = np.array(e.dxf.end)
            length = np.linalg.norm(end - start)
            y_avg = (start[1] + end[1]) / 2
            if abs(y_avg - y_min) < y_tol:
                if length > max_length:
                    max_length = length
                    main_line = e
    if main_line is None:
        raise Exception("未找到合适的主线（靠近下边界）")

    # 2. 获取WATERMARK图层上的所有短线
    watermark_lines = []
    for e in msp:
        if e.dxftype() == 'LINE' and e.dxf.layer == 'WATERMARK':
            start = np.array(e.dxf.start)
            end = np.array(e.dxf.end)
            watermark_lines.append((start, end))

    # 3. 主线等分
    start = np.array(main_line.dxf.start)
    end = np.array(main_line.dxf.end)
    vec = (end - start) / num_bits
    bits = []
    for i in range(num_bits):
        seg_start = start + i * vec
        seg_end = start + (i + 1) * vec
        center = (seg_start + seg_end) / 2
        radius = np.linalg.norm(seg_end[:2] - seg_start[:2]) / 2
        dir_vec = seg_end[:2] - seg_start[:2]
        dir_vec_unit = dir_vec / np.linalg.norm(dir_vec)
        norm_vec = np.array([-dir_vec[1], dir_vec[0]]) / np.linalg.norm(dir_vec)
        z = center[2]
        # 在该段主线上找短线中点
        found = False
        for line_start, line_end in watermark_lines:
            mid = (line_start + line_end) / 2
            if abs(mid[2] - z) > 1.0:
                continue
            if point_on_segment(mid[:2], seg_start[:2], seg_end[:2]):
                P_xy = mid[:2]
                # 过P作主线的垂线，与圆相交
                PO = P_xy - center[:2]
                a = 1
                b = 2 * np.dot(norm_vec, PO)
                c = np.dot(PO, PO) - radius**2
                delta = b**2 - 4*a*c
                if delta < 0:
                    continue
                t1 = (-b + np.sqrt(delta)) / (2*a)
                t2 = (-b - np.sqrt(delta)) / (2*a)
                t = t1 if abs(t1) > abs(t2) else t2
                Q_xy = P_xy + norm_vec * t
                A = seg_start[:2]
                B = seg_end[:2]
                Q = Q_xy
                alpha = angle_between(Q, A, B)
                beta = angle_between(Q, B, A)
                diff = abs(alpha - beta)
                if 0 <= diff < 30:
                    bits.append('0')
                elif 30 <= diff < 60:
                    bits.append('1')
                else:
                    bits.append('?')
                found = True
                break
        if not found:
            bits.append('?')
    return ''.join(bits)

def main():
    parser = argparse.ArgumentParser(description="DXF水印提取（主线分段+短线角度差值法，靠近下边界主线）")
    parser.add_argument('--input', type=str, required=True, help='输入DXF文件路径')
    parser.add_argument('--num_bits', type=int, required=True, help='要提取的水印比特数')
    args = parser.parse_args()

    watermark = extract_watermark_dxf(args.input, args.num_bits)
    print(f"提取的水印比特串为: {watermark}")

if __name__ == "__main__":
    main() 