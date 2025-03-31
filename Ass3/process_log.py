import re

def process_training_log(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 用于存储处理后的行
    processed_lines = []
    
    # 用于匹配各种进度行的模式
    progress_patterns = [
        r"Training:\s+\d+%.*\[\d+:\d+<.*it/s\]",
        r"Evaluating:\s+\d+%.*\[\d+:\d+<.*it/s\]",
        r"Calculating Perplexity:\s+\d+%.*\[\d+:\d+<.*it/s\]"
    ]
    
    # 用于匹配100%完成的行
    complete_patterns = [
        r"Training: 100%\|.*\| \d+/\d+ \[\d+:\d+<\d+:\d+,\s+\d+\.\d+it/s\]",
        r"Evaluating: 100%\|.*\| \d+/\d+ \[\d+:\d+<\d+:\d+,\s+\d+\.\d+it/s\]",
        r"Calculating Perplexity: 100%\|.*\| \d+/\d+ \[\d+:\d+<\d+:\d+,\s+\d+\.\d+it/s\]"
    ]
    
    # 用于临时存储当前epoch的100%进度行
    current_progress_lines = []
    in_progress_section = False
    
    for i, line in enumerate(lines):
        # 检查是否是进度行
        is_progress_line = any(re.match(pattern, line) for pattern in progress_patterns)
        
        if is_progress_line:
            # 检查是否是100%完成的行
            is_complete = any(re.match(pattern, line) for pattern in complete_patterns)
            
            if is_complete:
                current_progress_lines.append(line)
                in_progress_section = True
            continue
        
        # 如果遇到非进度行，且之前有进度行
        if in_progress_section and current_progress_lines:
            # 只保留最后一个100%的进度行
            processed_lines.append(current_progress_lines[-1])
            current_progress_lines = []
            in_progress_section = False
        
        # 保留非进度的行
        processed_lines.append(line)
    
    # 处理文件末尾可能的进度行
    if current_progress_lines:
        processed_lines.append(current_progress_lines[-1])
    
    # 写入处理后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)

if __name__ == "__main__":
    input_file = "/home/stu_12310401/nlp/SUSTech-NLP25/Ass3/A3_lm-lstm.txt"
    output_file = "/home/stu_12310401/nlp/SUSTech-NLP25/Ass3/A3_lm-lstm_processed.txt"
    
    process_training_log(input_file, output_file)
    print("处理完成！输出文件已保存为:", output_file)