This is a repository to extract x-vectors from the libritts dataset (https://research.google/tools/datasets/libri-tts/) from the espnet git repository. 

## Usage and Installation
### Installation of ParallelWaveGAN
```
git clone https://github.com/MeiGM/ParallelWaveGAN.git
cd ParallelWaveGAN
pip install -e .
```
``` 
pip install espnet
```
### Installation of necessary Packages for Kaldi install
```
mkdir -p espnet/tools/venv/bin
touch espnet/tools/venv/bin/activate
git clone https://github.com/espnet/warp-ctc -b pytorch-1.1
cd warp-ctc && mkdir build && cd build && cmake .. && make -j4
cd warp-ctc/pytorch_binding && python setup.py install
cd /espnet/tools
apt-get install -qq g++ automake autoconf libtool subversion
```
### Kaldi installation
```
git clone https://github.com/kaldi-asr/kaldi
os.chdir('kaldi//tools')
bash extras//install_mkl.sh
bash extras/install_irstlm.sh
sudo apt-get install sox
```
```
bash extras//check_dependencies.sh
```
In case the system has all the required packages the check_dependencies.sh file will output ALL OK. Else it will enlist the missing packages. Rerun the file after installiing all the requirements. 
```make -j 2
cd espnet//tools//kaldi//src
./configure
make depend -j 4
make -j 4
```
## Extracting X-vectors 
Once all the necessary packages have been installed:
* Change MAIN_ROOT and KALDI_ROOT in path.sh file in espnet//egs//libritts//tts1.
* Run the x-vector extraction file, run.sh in libritts/tts1 folder.
``` 
cd  espnet//egs//libritts//tts1
bash run.sh
```
## Seaprating X-vectors for different speakers from the .ark file
Run the xvec_extraction.py file, it will generate a new folder (xvectors) in the current working directory with the embedding file for each speaker. 

Note: Incase the directory of the .ark files is changed from exp/xvector_nnet_1a/xvectors_train_clean_100, make changes accordingly in the xvec_extraction.py file
## References

[1] Shinji Watanabe, Takaaki Hori, Shigeki Karita, Tomoki Hayashi, Jiro Nishitoba, Yuya Unno, Nelson Enrique Yalta Soplin, Jahn Heymann, Matthew Wiesner, Nanxin Chen, Adithya Renduchintala, and Tsubasa Ochiai, "ESPnet: End-to-End Speech Processing Toolkit," *Proc. Interspeech'18*, pp. 2207-2211 (2018)

[2] Suyoun Kim, Takaaki Hori, and Shinji Watanabe, "Joint CTC-attention based end-to-end speech recognition using multi-task learning," *Proc. ICASSP'17*, pp. 4835--4839 (2017)

[3] Shinji Watanabe, Takaaki Hori, Suyoun Kim, John R. Hershey and Tomoki Hayashi, "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition," *IEEE Journal of Selected Topics in Signal Processing*, vol. 11, no. 8, pp. 1240-1253, Dec. 2017

## Citations

```
@inproceedings{watanabe2018espnet,
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson {Enrique Yalta Soplin} and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  title={{ESPnet}: End-to-End Speech Processing Toolkit},
  year={2018},
  booktitle={Proceedings of Interspeech},
  pages={2207--2211},
  doi={10.21437/Interspeech.2018-1456},
  url={http://dx.doi.org/10.21437/Interspeech.2018-1456}
}
@inproceedings{hayashi2020espnet,
  title={{Espnet-TTS}: Unified, reproducible, and integratable open source end-to-end text-to-speech toolkit},
  author={Hayashi, Tomoki and Yamamoto, Ryuichi and Inoue, Katsuki and Yoshimura, Takenori and Watanabe, Shinji and Toda, Tomoki and Takeda, Kazuya and Zhang, Yu and Tan, Xu},
  booktitle={Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7654--7658},
  year={2020},
  organization={IEEE}
}
@inproceedings{inaguma-etal-2020-espnet,
    title = "{ESP}net-{ST}: All-in-One Speech Translation Toolkit",
    author = "Inaguma, Hirofumi  and
      Kiyono, Shun  and
      Duh, Kevin  and
      Karita, Shigeki  and
      Yalta, Nelson  and
      Hayashi, Tomoki  and
      Watanabe, Shinji",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-demos.34",
    pages = "302--311",
}
```
