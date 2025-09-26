# mydataset 数据集类，用来读取.wav数据，并提取特征
from pyexpat import features
import sys
import os
from datetime import datetime
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from scipy.fftpack import dct  # 在文件顶部添加导入

def load_audio(audio_path, sr=16000, chunk_duration=3, features=['mel', 'gfcc', 'stft', 'cqt']):
    """
    音频特征提取函数 | Audio feature extraction
    
    新增可选特征:
    4. GFCC (Gammatone Frequency Cepstral Coefficients)
    5. STFT (Short-Time Fourier Transform)
    6. CQT (Constant-Q Transform)
    
    Args:
        audio_path: 音频文件路径
        sr: 采样率 (默认16000)
        chunk_duration: 截取时长(秒)
        features: 要提取的特征列表，可选'mel', 'welch', 'avg', 'gfcc', 'stft', 'cqt'
    """
    wav, sr_ret = librosa.load(audio_path, sr=sr)
    num_wav_samples = wav.shape[0]
    num_chunk_samples = int(chunk_duration * sr)
    if num_wav_samples > num_chunk_samples:
        wav = wav[:num_chunk_samples]
    
    result = {}
    
    if 'mel' in features:
        # 计算Mel频谱
        mel_spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=1024, n_mels=200, hop_length=160, win_length=400)
        result['mel'] = librosa.power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=None)

    if 'welch' in features:
        # 计算韦尔奇谱
        freqs, welch_spectrum = signal.welch(wav, fs=sr, nperseg=400)
        result['welch'] = 10 * np.log10(welch_spectrum)

    if 'avg' in features:
        # 计算平均幅度谱
        avg_amp_spectrum = np.abs(np.fft.rfft(wav, n=400))
        result['avg'] = 10 * np.log10(avg_amp_spectrum+1)

    if 'gfcc' in features:
        # 计算GFCC
        gammatone = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=200, fmin=50, fmax=8000)
        gfcc = dct(10 * np.log10(gammatone + 1e-10), type=2, axis=0, norm='ortho')  # 使用scipy的dct
        result['gfcc'] = gfcc  # 取前13个系数

    if 'stft' in features:
        # 计算STFT
        stft = librosa.stft(wav, n_fft=1024, hop_length=160)
        result['stft'] = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    if 'cqt' in features:
        # 计算CQT
        cqt = librosa.cqt(wav, sr=sr, fmin=librosa.note_to_hz('C1'), bins_per_octave=24)
        result['cqt'] = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    if 'mfcc' in features:
        # 计算MFCC（梅尔频率倒谱系数）
        mfcc = librosa.feature.mfcc(
            y=wav, 
            sr=sr,
            n_mfcc=40,
            n_fft=1024,
            hop_length=160,
            n_mels=200
        )
        result['mfcc'] = librosa.util.normalize(mfcc, axis=1)

    return result

class Dataset_audio(Dataset):
    """
    音频数据集类 | Audio dataset class
    
    特征缓存机制：
    1. 优先加载预计算的特征文件（.npy）
    2. 特征缺失时实时计算并保存
    3. 异常时自动重试随机样本
    
    Feature caching mechanism:
    1. Load pre-computed features first
    2. Compute and save when missing
    3. Auto-retry with random sample on error
    """
    def __init__(self, data_list_path, sr=16000, chunk_duration=5, features=['mel', 'gfcc', 'stft', 'cqt', 'mfcc']):
        super(Dataset_audio, self).__init__()
        with open(data_list_path, 'r') as f:
            self.lines = f.readlines()
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.features = features

    def __getitem__(self, idx):
        try:
            audio_path, label, Rr, Sz = self.lines[idx].replace('\n', '').split('\t')
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            
            # 特征缓存路径和版本控制
            feature_dir = os.path.join(os.path.dirname(audio_path), 'features')
            cache_version = 'v1'  # 当特征提取参数变更时更新版本号
            
            # 初始化特征字典和缺失标志
            features = {}
            cache_missing = False
            
            # 检查所有特征缓存是否存在
            for feat_name in self.features:
                feat_path = os.path.join(feature_dir, f"{base_name}_{feat_name}_{cache_version}.npy")
                if not os.path.exists(feat_path):
                    cache_missing = True
                    break
            
            if not cache_missing:
                # 加载所有缓存特征
                for feat_name in self.features:
                    feat_path = os.path.join(feature_dir, f"{base_name}_{feat_name}_{cache_version}.npy")
                    features[feat_name] = np.load(feat_path)
            else:
                # 计算并保存新特征
                features = load_audio(audio_path, sr=self.sr, chunk_duration=self.chunk_duration, features=self.features)
                if not os.path.exists(feature_dir):
                    os.makedirs(feature_dir)
                for feat_name, feat_data in features.items():
                    feat_path = os.path.join(feature_dir, f"{base_name}_{feat_name}_{cache_version}.npy")
                    np.save(feat_path, feat_data.astype('float32'))
            
            # 返回特征和标签
            return (features, 
                    np.array(int(label), dtype=np.int64), 
                    np.array(Rr, dtype=np.float64), 
                    np.array(Sz, dtype=np.float64))
        except Exception as ex:
            print(f"[{datetime.now()}] 数据: {self.lines[idx]} 出错，错误信息: {ex}", file=sys.stderr)
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.lines)

# 使用示例
if __name__ == "__main__":
    data_list_path = r"E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_test_list.txt"
    dataset = Dataset_audio(data_list_path, features=['mel', 'gfcc', 'stft', 'cqt','mfcc'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for features, label, Rr, Rz in dataloader:
        # 打印
        for feat_name, feat_data in features.items():
            print(f"{feat_name}: {feat_data.shape}")
