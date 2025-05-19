import ezdxf
import numpy as np
import argparse
from sklearn.cluster import KMeans


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


def embed_watermark_dxf(input_path, output_path, watermark_bits, clusters=3, segments=8, search_steps=100):
    doc = ezdxf.readfile(input_path)
    msp = doc.modelspace()

    # 先遍历一次，获取图纸宽度
    xs = []
    for e in msp:
        if e.dxftype() == 'LINE':
            xs.extend([e.dxf.start[0], e.dxf.end[0]])
    x_min, x_max = min(xs), max(xs)
    drawing_width = x_max - x_min

    # 然后筛选主线
    lines = []
    lengths = []
    min_length = drawing_width * 0.05  # 只保留长度大于图纸宽度30%的主线
    for e in msp:
        if e.dxftype() == 'LINE' and is_horizontal(e):
            start = np.array(e.dxf.start)
            end = np.array(e.dxf.end)
            length = np.linalg.norm(end - start)
            if length > min_length:
                lines.append(e)
                lengths.append([length])
    if len(lines) < clusters:
        raise Exception(f"水平主线数量不足，无法聚类为{clusters}个簇")

    # 2. 聚类
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(lengths)
    labels = kmeans.labels_

    # 3. 每簇内选中位数主线
    main_lines = []
    for k in range(clusters):
        cluster_lines = []
        for idx, e in enumerate(lines):
            if labels[idx] == k:
                start = np.array(e.dxf.start)
                end = np.array(e.dxf.end)
                length = np.linalg.norm(end - start)
                cluster_lines.append((e, length))
        if cluster_lines:
            cluster_lines.sort(key=lambda x: x[1])
            mid_idx = len(cluster_lines) // 2
            main_line = cluster_lines[mid_idx][0]
            main_lines.append(main_line)

    # 4. 按长度降序排序，保证顺序一致
    main_lines = sorted(main_lines, key=lambda e: -np.linalg.norm(np.array(e.dxf.end) - np.array(e.dxf.start)))

    # 5. 检查水印长度
    total_bits = clusters * segments
    if len(watermark_bits) != total_bits:
        raise Exception(f"水印比特串长度应为{total_bits}，实际为{len(watermark_bits)}")

    # 6. 对每条主线等分嵌入
    bit_idx = 0
    for main_line in main_lines:
        start = np.array(main_line.dxf.start)
        end = np.array(main_line.dxf.end)
        vec = (end - start) / segments
        for i in range(segments):
            seg_start = start + i * vec
            seg_end = start + (i + 1) * vec
            center = (seg_start + seg_end) / 2
            radius = np.linalg.norm(seg_end[:2] - seg_start[:2]) / 2
            dir_vec = seg_end[:2] - seg_start[:2]
            dir_vec_unit = dir_vec / np.linalg.norm(dir_vec)
            norm_vec = np.array([-dir_vec[1], dir_vec[0]]) / np.linalg.norm(dir_vec)
            z = center[2]
            fixed_length = 0.002  # 短线总长度
            short_line_half_len = fixed_length / 2
            bit = watermark_bits[bit_idx]
            found = False
            for offset in np.linspace(-radius, radius, search_steps):
                P_xy = center[:2] + dir_vec_unit * offset
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
                    perp_vec = norm_vec
                    line_start = P_xy - perp_vec * short_line_half_len
                    line_end = P_xy + perp_vec * short_line_half_len
                    msp.add_line((line_start[0], line_start[1], z), (line_end[0], line_end[1], z), dxfattribs={'layer': 'WATERMARK'})
                    found = True
                    break
            if not found:
                print(f"警告：主线{main_line.dxf.handle} 第{i+1}段未找到满足条件的嵌入点，跳过")
            bit_idx += 1

    doc.saveas(output_path)
    print(f"水印嵌入完成，结果已保存到 {output_path}")


def main():
    parser = argparse.ArgumentParser(description="自适应聚类多主线水印嵌入（画圆+角度差值法）")
    parser.add_argument('--input', type=str, required=True, help='输入DXF文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出带水印DXF文件路径')
    parser.add_argument('--watermark', type=str, required=True, help='水印比特串，如10101010...')
    parser.add_argument('--clusters', type=int, default=3, help='主线聚类数（主线数）')
    parser.add_argument('--segments', type=int, default=8, help='每条主线分段数')
    args = parser.parse_args()

    embed_watermark_dxf(args.input, args.output, args.watermark, args.clusters, args.segments)

if __name__ == "__main__":
    main() 