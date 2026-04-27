import os
import json
import ast

def svg_lineplot(out_path, title, ylabel, xlabel, data_dict, y_max_cap=None):
    """
    手写一个纯 Python 无依赖的 SVG 折线图生成器
    """
    width, height = 980, 560
    # 留出右侧空间放图例 (Legend)
    left, right, top, bottom = 90, 180, 70, 70 
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_x, all_y = [], []
    for name, (xs, ys) in data_dict.items():
        all_x.extend(xs)
        if y_max_cap:
            all_y.extend([min(y, y_max_cap) for y in ys])
        else:
            all_y.extend(ys)

    if not all_x or not all_y:
        print(f"[提示] 没有数据用于绘制: {title}")
        return

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    if max_x == min_x: max_x = min_x + 1
    if max_y == min_y: max_y = min_y + 1

    if y_max_cap and max_y > y_max_cap:
        max_y = y_max_cap

    def get_cx(x): return left + (x - min_x) / (max_x - min_x) * plot_w
    def get_cy(y): return top + plot_h - (min(y, max_y) - min_y) / (max_y - min_y) * plot_h

    # 给5个模型分配的颜色表
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#EECA3B", "#B279A2"]

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    parts.append(f'<text x="{width/2}" y="32" text-anchor="middle" font-size="22" font-family="Arial, sans-serif" font-weight="bold">{title}</text>')

    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333333" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333333" stroke-width="2"/>')

    num_ticks = 5
    for i in range(num_ticks + 1):
        v = min_y + (max_y - min_y) * i / num_ticks
        y = get_cy(v)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#eeeeee" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" font-size="12" font-family="Arial">{v:.2f}</text>')

    for i in range(num_ticks + 1):
        v = min_x + (max_x - min_x) * i / num_ticks
        x = get_cx(v)
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+20}" text-anchor="middle" font-size="12" font-family="Arial">{int(v)}</text>')

    parts.append(f'<text x="24" y="{top + plot_h/2}" transform="rotate(-90 24 {top + plot_h/2})" text-anchor="middle" font-size="16" font-family="Arial">{ylabel}</text>')
    parts.append(f'<text x="{left + plot_w/2}" y="{top + plot_h + 45}" text-anchor="middle" font-size="16" font-family="Arial">{xlabel}</text>')

    for idx, (name, (xs, ys)) in enumerate(data_dict.items()):
        color = colors[idx % len(colors)]
        
        points_str = " ".join([f"{get_cx(x):.1f},{get_cy(y):.1f}" for x, y in zip(xs, ys)])
        parts.append(f'<polyline points="{points_str}" fill="none" stroke="{color}" stroke-width="2.5" opacity="0.85"/>')

        # 给所有点画上圆圈，因为现在采样变稀疏了，画圆圈会很好看
        for x, y in zip(xs, ys):
            parts.append(f'<circle cx="{get_cx(x):.1f}" cy="{get_cy(y):.1f}" r="3" fill="{color}"/>')

        leg_x = left + plot_w + 20
        leg_y = top + 20 + idx * 30
        parts.append(f'<line x1="{leg_x}" y1="{leg_y-5}" x2="{leg_x+25}" y2="{leg_y-5}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{leg_x+35}" y="{leg_y}" font-size="14" font-family="Arial" fill="#333333">{name}</text>')

    parts.append('</svg>')

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"[成功] 图表已保存为 {out_path}")

# ==========================================
# 主流程：读取数据并绘图
# ==========================================
if __name__ == "__main__":
    # 配置你的模型路径
    model_versions = {
        "Base_Model_1": "./logs/unisyn_base_2",
        "Base_Model_2": "./logs/unisyn_base_3",
        "Base_Model_3": "./logs/unisyn_base_4",
        "Base_Model_4": "./logs/unisyn_base_5",
        "Base_Model_5": "./logs/unisyn_base_6"
    }

    train_data = {}
    val_data = {}
    
    # 自动换算系数：1个Epoch大约等于816步 (你可以根据实际情况微调)
    STEPS_PER_EPOCH = 816.0 

    for label, log_dir in model_versions.items():
        # --- 1. 读取 Training Loss (按 Epoch 稀疏化) ---
        log_path = os.path.join(log_dir, "train.log")
        if os.path.exists(log_path):
            epoch_loss_dict = {} # 用于存储 {epoch_num: loss}
            current_epoch = 0
            
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 抓取当前的 Epoch 轮数
                    if 'Train Epoch:' in line:
                        try:
                            # 提取 "Train Epoch: 1 [98%]" 里面的 "1"
                            current_epoch = int(line.split('Train Epoch:')[1].split('[')[0].strip())
                        except:
                            pass
                            
                    # 抓取 Loss 数组
                    elif 'INFO\t[' in line:
                        try:
                            list_str = line.split('INFO\t')[-1].strip()
                            data_list = ast.literal_eval(list_str)
                            loss_val = data_list[1] # loss_gen_all
                            
                            # 关键：不断覆盖当前 epoch 的字典值，最后存下来的就是该 epoch 最末尾的一步
                            if current_epoch > 0:
                                epoch_loss_dict[current_epoch] = loss_val
                        except:
                            pass
                            
            if epoch_loss_dict:
                # 排序整理成绘图用的列表
                t_epochs = sorted(epoch_loss_dict.keys())
                t_losses = [epoch_loss_dict[ep] for ep in t_epochs]
                train_data[label] = (t_epochs, t_losses)

        # --- 2. 读取 Validation Loss ---
        jsonl_path = os.path.join(log_dir, "eval_metrics.jsonl")
        if os.path.exists(jsonl_path):
            v_epochs, v_losses = [], []
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        step = data['global_step']
                        # Validation 只有 Step，我们把它除以每轮的步数，换算成带小数点的 Epoch
                        epoch_val = step / STEPS_PER_EPOCH
                        v_epochs.append(epoch_val)
                        v_losses.append(data['val_mel_loss'])
                    except:
                        pass
            if v_epochs:
                val_data[label] = (v_epochs, v_losses)

    # 绘制 SVG (X轴全部变为 Epoch)
    svg_lineplot("training_loss_sparse.svg", "UniSyn Training Loss", "Loss", "Epoch", train_data, y_max_cap=300)
    svg_lineplot("validation_loss.svg", "UniSyn Validation Loss", "Val Mel Loss", "Epoch", val_data)