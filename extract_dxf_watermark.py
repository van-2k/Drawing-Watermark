import ezdxf
import numpy as np
import argparse
from sklearn.cluster import KMeans

def get_line_angle(line):
    """获取线段与x轴的夹角（-180到180度）"""
    start = np.array(line.dxf.start)
    end = np.array(line.dxf.end)
    vec = end - start
    return np.degrees(np.arctan2(vec[1], vec[0]))

def get_dominant_angle(lines):
    """获取图纸的主要方向角度"""
    angles = []
    for line in lines:
        angle = get_line_angle(line)
        angles.append(angle)
    
    # 将角度转换到0-180度范围
    angles = [angle % 180 for angle in angles]
    
    # 使用直方图统计最频繁的角度
    hist, bins = np.histogram(angles, bins=180, range=(0, 180))
    dominant_angle = bins[np.argmax(hist)]
    
    return dominant_angle

def rotate_point(point, center, angle_degrees):
    """绕中心点旋转点"""
    angle_rad = np.radians(angle_degrees)
    x, y = point[0] - center[0], point[1] - center[1]
    cos_theta, sin_theta = np.cos(angle_rad), np.sin(angle_rad)
    x_new = x * cos_theta - y * sin_theta
    y_new = x * sin_theta + y * cos_theta
    return np.array([x_new + center[0], y_new + center[1]])

def angle_between(a, b, c):
    """计算∠abc的度数"""
    ba = a - b
    bc = c - b
    # 检查向量长度是否为0
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm == 0 or bc_norm == 0:
        return 0
    cos_angle = np.dot(ba, bc) / (ba_norm * bc_norm)
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    return np.degrees(angle)

def point_on_segment(p, a, b):
    """判断点p是否在线段ab上"""
    pa = p - a
    pb = p - b
    ab = b - a
    # 检查点是否在线段范围内
    if np.dot(pa, ab) < 0 or np.dot(pb, -ab) < 0:
        return False
    # 检查点是否在线段上
    cross = np.cross(pa, ab)
    return abs(cross) < 1e-10

def find_watermark_lines(msp, search_radius=0.003):
    """查找水印短线"""
    watermark_lines = []
    for e in msp:
        if e.dxftype() == 'LINE':
            start = np.array(e.dxf.start)
            end = np.array(e.dxf.end)
            # 确保使用3D向量
            if len(start) == 2:
                start = np.append(start, 0)
            if len(end) == 2:
                end = np.append(end, 0)
            length = np.linalg.norm(end - start)
            # 水印短线长度约为0.002
            if 0.0005 <= length <= 0.06:
                watermark_lines.append(e)
    return watermark_lines

def find_main_lines(msp, watermark_lines, line_min_length, search_radius=0.003):
    """查找可能嵌入水印的主线"""
    main_lines = []
    for e in msp:
        if e.dxftype() == 'LINE':
            start = np.array(e.dxf.start)
            end = np.array(e.dxf.end)
            # 确保使用3D向量
            if len(start) == 2:
                start = np.append(start, 0)
            if len(end) == 2:
                end = np.append(end, 0)
            length = np.linalg.norm(end - start)
            if length >= line_min_length:
                # 检查这条线段附近是否有水印线
                has_watermark = False
                for wl in watermark_lines:
                    wl_start = np.array(wl.dxf.start)
                    wl_end = np.array(wl.dxf.end)
                    # 确保使用3D向量
                    if len(wl_start) == 2:
                        wl_start = np.append(wl_start, 0)
                    if len(wl_end) == 2:
                        wl_end = np.append(wl_end, 0)
                    wl_center = (wl_start + wl_end) / 2
                    # 计算水印线中心到主线的距离
                    dist = np.abs(np.cross(end - start, wl_center - start)[2]) / np.linalg.norm(end - start)
                    if dist <= search_radius:
                        has_watermark = True
                        break
                if has_watermark:
                    main_lines.append(e)
    return main_lines

def extract_watermark_dxf(input_path, clusters=3, segments=8):
    doc = ezdxf.readfile(input_path)
    msp = doc.modelspace()

    # 1. 计算图纸尺寸
    xs = []
    ys = []
    for e in msp:
        if e.dxftype() == 'LINE':
            xs.extend([e.dxf.start[0], e.dxf.end[0]])
            ys.extend([e.dxf.start[1], e.dxf.end[1]])
    
    if not xs or not ys:
        raise Exception("图纸中没有线段")
    
    drawing_width = max(xs) - min(xs)
    drawing_height = max(ys) - min(ys)
    drawing_size = max(drawing_width, drawing_height)
    
    # 设置线段最小长度阈值（与嵌入算法保持一致）
    line_min_length = drawing_size * 0.02
    print(f"图纸尺寸: {drawing_size:.3f}")
    print(f"最小线段长度阈值: {line_min_length:.3f}")

    # 2. 获取所有符合条件的线段
    lines = []
    lengths = []
    for e in msp:
        if e.dxftype() == 'LINE':
            start = np.array(e.dxf.start)
            end = np.array(e.dxf.end)
            # 确保使用3D向量
            if len(start) == 2:
                start = np.append(start, 0)
            if len(end) == 2:
                end = np.append(end, 0)
            length = np.linalg.norm(end - start)
            # 只选择长度大于阈值的线段
            if length >= line_min_length:
                lines.append(e)
                lengths.append([length])
                # print(f"找到长线段: {e.dxf.handle}, 长度: {length:.3f}")

    if len(lines) < clusters:
        raise Exception(f"线段数量不足，无法聚类为{clusters}个簇")

    # 3. 根据长度聚类
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(lengths)
    labels = kmeans.labels_

    # 4. 每簇内选中位数线段
    selected_lines = []
    for k in range(clusters):
        cluster_lines = []
        for idx, e in enumerate(lines):
            if labels[idx] == k:
                start = np.array(e.dxf.start)
                end = np.array(e.dxf.end)
                # 确保使用3D向量
                if len(start) == 2:
                    start = np.append(start, 0)
                if len(end) == 2:
                    end = np.append(end, 0)
                length = np.linalg.norm(end - start)
                cluster_lines.append((e, length))
        if cluster_lines:
            cluster_lines.sort(key=lambda x: x[1])
            mid_idx = len(cluster_lines) // 2
            mid_length = cluster_lines[mid_idx][1]
            # 找到所有长度等于中位数的线段
            candidates = [e for e, l in cluster_lines if np.isclose(l, mid_length)]
            # 选择 handle 最大的那一条
            selected_line = max(candidates, key=lambda e: int(e.dxf.handle, 16))
            selected_lines.append(selected_line)
            print(f"簇 {k} 选中线段: {selected_line.dxf.handle}, 长度: {mid_length:.3f}")

    # 5. 按长度降序排序
    selected_lines = sorted(selected_lines, key=lambda e: -np.linalg.norm(np.array(e.dxf.end) - np.array(e.dxf.start)))
    # print("\n最终选中的线段:")
    for line in selected_lines:
        start = np.array(line.dxf.start)
        end = np.array(line.dxf.end)
        length = np.linalg.norm(end - start)
        # print(f"线段 {line.dxf.handle}:")
        # print(f"  起点: ({start[0]:.3f}, {start[1]:.3f})")
        # print(f"  终点: ({end[0]:.3f}, {end[1]:.3f})")
        # print(f"  长度: {length:.3f}")

    # 6. 查找所有水印短线
    watermark_lines = []
    for e in msp:
        if e.dxftype() == 'LINE':
            start = np.array(e.dxf.start)
            end = np.array(e.dxf.end)
            # 确保使用3D向量
            if len(start) == 2:
                start = np.append(start, 0)
            if len(end) == 2:
                end = np.append(end, 0)
            length = np.linalg.norm(end - start)
            # 水印短线长度约为0.002
            if 0.0005 <= length <= 0.06:
                watermark_lines.append(e)
    #             print(f"找到水印短线: {e.dxf.handle}")
    #             print(f"  起点: ({start[0]:.3f}, {start[1]:.3f})")
    #             print(f"  终点: ({end[0]:.3f}, {end[1]:.3f})")
    #             print(f"  长度: {length:.3f}")

    # print(f"\n找到 {len(watermark_lines)} 条水印短线")

    # 7. 提取水印
    watermark_bits = []
    for line in selected_lines:
        start = np.array(line.dxf.start)
        end = np.array(line.dxf.end)
        # 确保使用3D向量
        if len(start) == 2:
            start = np.append(start, 0)
        if len(end) == 2:
            end = np.append(end, 0)
        
        # 计算主线方向向量
        main_vec = end - start
        main_vec = main_vec / np.linalg.norm(main_vec)
        
        # 对线段进行等分
        vec = (end - start) / segments
        for i in range(segments):
            seg_start = start + i * vec
            seg_end = start + (i + 1) * vec
            center_seg = (seg_start + seg_end) / 2
            
            # 查找这个分段附近的水印线
            segment_watermarks = []
            for wl in watermark_lines:
                wl_start = np.array(wl.dxf.start)
                wl_end = np.array(wl.dxf.end)
                # 确保使用3D向量
                if len(wl_start) == 2:
                    wl_start = np.append(wl_start, 0)
                if len(wl_end) == 2:
                    wl_end = np.append(wl_end, 0)
                wl_center = (wl_start + wl_end) / 2
                
                # 计算水印线方向向量
                wl_vec = wl_end - wl_start
                wl_vec = wl_vec / np.linalg.norm(wl_vec)
                
                # 检查水印线是否在这个分段附近
                dist = np.abs(np.cross(end - start, wl_center - start)[2]) / np.linalg.norm(end - start)
                if dist <= 0.1:  # 进一步放宽搜索半径
                    # 计算水印线中心到分段中心的距离
                    dist_to_center = np.linalg.norm(wl_center - center_seg)
                    if dist_to_center <= np.linalg.norm(vec) * 1.5:  # 进一步放宽分段范围
                        # 检查水印线是否垂直于主线
                        # 使用叉积判断垂直关系
                        cross_product = np.abs(np.cross(main_vec, wl_vec)[2])
                        if cross_product > 0.9:  # 允许夹角在约±30°范围内
                            # 计算角度差值
                            # 以分段为直径的圆上的点
                            radius = np.linalg.norm(vec) / 2
                            # 计算水印线延长线与圆的交点
                            # 使用参数方程：p = wl_center + t * wl_vec
                            # 与圆的方程：(x - center_seg[0])^2 + (y - center_seg[1])^2 = radius^2
                            # 求解t
                            a = np.dot(wl_vec, wl_vec)
                            b = 2 * np.dot(wl_vec, wl_center - center_seg)
                            c = np.dot(wl_center - center_seg, wl_center - center_seg) - radius * radius
                            discriminant = b * b - 4 * a * c
                            if discriminant >= 0:
                                t1 = (-b + np.sqrt(discriminant)) / (2 * a)
                                t2 = (-b - np.sqrt(discriminant)) / (2 * a)
                                # 选择更远的交点
                                t = t1 if abs(t1) > abs(t2) else t2
                                intersection = wl_center + t * wl_vec
                                
                                # 计算角度差值
                                alpha = angle_between(intersection, seg_start, seg_end)
                                beta = angle_between(intersection, seg_end, seg_start)
                                diff = abs(alpha - beta)
                                segment_watermarks.append((diff, wl_center))
                                # print(f"找到水印线: 线段{line.dxf.handle} 第{i+1}段")
                                # print(f"  水印线中心: ({wl_center[0]:.3f}, {wl_center[1]:.3f})")
                                # print(f"  到分段中心距离: {dist_to_center:.3f}")
                                # print(f"  到主线距离: {dist:.3f}")
                                # print(f"  叉积: {cross_product:.3f}")
                                # print(f"  角度差值: {diff:.3f}")
            
            if segment_watermarks:
                # 找到最近的水印线
                segment_watermarks.sort(key=lambda x: np.linalg.norm(x[1] - center_seg))
                diff = segment_watermarks[0][0]
                
                # 根据角度差值判断水印比特
                if diff < 30:  # 0比特的角度差值在0-30度之间
                    watermark_bits.append('0')
                elif diff < 60:  # 1比特的角度差值在30-60度之间
                    watermark_bits.append('1')
                else:
                    # 如果角度差值超出范围，使用默认值
                    watermark_bits.append('0')
                    # print(f"警告：线段{line.dxf.handle} 第{i+1}段角度差值超出范围：{diff}")
            else:
                # 如果没有找到水印线，使用默认值
                watermark_bits.append('0')
                # print(f"警告：线段{line.dxf.handle} 第{i+1}段未找到水印线")

    return ''.join(watermark_bits)

def main():
    parser = argparse.ArgumentParser(description="自适应聚类水印提取（画圆+角度差值法）")
    parser.add_argument('--input', type=str, required=True, help='输入带水印DXF文件路径')
    parser.add_argument('--clusters', type=int, default=3, help='线段聚类数')
    parser.add_argument('--segments', type=int, default=8, help='每条线段分段数')
    args = parser.parse_args()

    watermark = extract_watermark_dxf(args.input, args.clusters, args.segments)
    print(f"提取的水印比特串：{watermark}")

if __name__ == "__main__":
    main()