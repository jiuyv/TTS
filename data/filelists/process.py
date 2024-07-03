dir = [r"data\filelists\ljs_audio_text_train_filelist.txt",
       r"data\filelists\ljs_audio_text_val_filelist.txt",
       r"data\filelists\ljs_audio_text_test_filelist.txt"]

for d in dir:
    print("Processing", d)
    with open(d, "r", encoding='utf-8') as file:
        lines = file.readlines()

    # 替换路径
    lines = [line.replace(r"/root/TTS/LJSpeech/wavs/LJ", r"D:\Code\Experiment\Python\TTS\LJSpeech\wavs\LJ") for line in lines]

    # 写回文件
    with open(d, "w", encoding='utf-8') as file:
        file.writelines(lines)