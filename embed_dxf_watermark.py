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


def embed_watermark_dxf(input_path, output_path, watermark_bits, search_steps=100):
    doc = ezdxf.readfile(input_path)
    msp = doc.modelspace()

    # 1. 获取图纸宽度和y范围
    xs = []
    ys = []
    for e in msp:
        if e.dxftype() == 'LINE':
            xs.extend([e.dxf.start[0], e.dxf.end[0]])
            ys.extend([e.dxf.start[1], e.dxf.end[1]])
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    drawing_width = x_max - x_min

    # 2. 只考虑靠近y_min的最长水平主线
    main_line = None
    max_length = 0
    y_tol = 1.0  # 容差，可根据实际调整
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

    # 3. 等分主线
    m = len(watermark_bits)
    start = np.array(main_line.dxf.start)
    end = np.array(main_line.dxf.end)
    vec = (end - start) / m

    for i, bit in enumerate(watermark_bits):
        seg_start = start + i * vec
        seg_end = start + (i + 1) * vec
        center = (seg_start + seg_end) / 2
        radius = np.linalg.norm(seg_end[:2] - seg_start[:2]) / 2
        dir_vec = seg_end[:2] - seg_start[:2]
        dir_vec_unit = dir_vec / np.linalg.norm(dir_vec)
        norm_vec = np.array([-dir_vec[1], dir_vec[0]]) / np.linalg.norm(dir_vec)
        z = center[2]
        seg_len = np.linalg.norm(seg_end[:2] - seg_start[:2])
        fixed_length = 0.1  # 你想要的短线总长度
        short_line_half_len = fixed_length / 2

        # 在主线上采样若干点P，判断角度差值
        found = False
        for offset in np.linspace(-radius, radius, search_steps):
            P_xy = center[:2] + dir_vec_unit * offset  # P_xy始终在主线上
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
            if (bit == '0' and 0 <= diff < 30) or (bit == '1' and 30 <= diff < 60):
                # 以P_xy为中点，沿norm_vec方向作短线
                perp_vec = norm_vec  # 单位法线
                line_start = P_xy - perp_vec * short_line_half_len
                line_end = P_xy + perp_vec * short_line_half_len
                msp.add_line((line_start[0], line_start[1], z), (line_end[0], line_end[1], z), dxfattribs={'layer': 'WATERMARK'})
                found = True
                break
        if not found:
            print(f"警告：第{i+1}段未找到满足条件的嵌入点，跳过")

    doc.saveas(output_path)
    print(f"水印嵌入完成，结果已保存到 {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DXF主线分段水印嵌入（画圆+角度差值法，靠近下边界主线）")
    parser.add_argument('--input', type=str, required=True, help='输入DXF文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出带水印DXF文件路径')
    parser.add_argument('--watermark', type=str, required=True, help='水印比特串，如10101010')
    args = parser.parse_args()

    embed_watermark_dxf(args.input, args.output, args.watermark)

if __name__ == "__main__":
    main() 