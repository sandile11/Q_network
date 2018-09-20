(Japanese below)

#Launching Method
1. Launch Fighting ICE with the start option "- py 4 j".
2. Run Sample_multi-head/Run_Sample_DQN.py.

#File Description
1. MultiHead.py in this foloder was trained for 1250 rounds against all of the participating nine AIs in the 2017 competition.
2. If you run Sample_multi-head/Run_Sample_DQN.py as it is, the trained weight parameters for ZEN will be used and a match between MultiHead and Machete will start.
3. You can change to the trained weight parameters for GARNET by modifying the 25th line of Sample_multi-head/MultiHead.py accordingly.

#Recommended Execution Environment
1. Linux
2. python 3.6

#Used Deep Learning Library
Pytorch

#Pytorch
Package Version
torchvision 0.1.6
torch 0.3.1

#How to Install Packages
1. Execute the following commands using pip
pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
pip install http://download.pytorch.org/whl/torchvision-0.1.6-py3-none-any.whl

2. Please see the details
https://pytorch.org/previous-versions/

#Related Paper
Yoshina Takano, Wenwen Ouyangy, Suguru Ito, Tomohiro Harada and Ruck Thawonmas, "Applying Hybrid Reward Architecture to a Fighting Game AI," 2018 IEEE Conference on Computational Intelligence and Games (CIG 2018), Maastricht, The Netherlands, Aug.14-17. 2018.
http://www.ice.ci.ritsumei.ac.jp/~ruck/PAP/cig2018-takano.pdf

#Related Video
https://youtu.be/9tPdGveY4CA
---------------------------------

・起動方法
1 起動オプション"--py4j"を付けてFightingICEを起動.
2 Sample_multi-head/Run_Sample_DQN.py を起動する.

・ファイル説明
1 開会版のMultiHead.pyは2017年の全ての出場AI(9 agents)に対して1250roundsトレーニングされている.
2 そのままSample_multi-head/Run_Sample_DQN.pyを実行するとZENのWeightをロードし，MultiHeadとMacheteの対戦が始まります．
3 Sample_multi-head/MultiHead.pyの25行目を変更するとGARNETのweightをロードする(ソースコード内にコメントとして書いています).

・実行推奨環境
1 Linux
2 python3.6

・使用深層学習用ライブラリ
Pytorch

・Pytorch 
Package       Version
torchvision	  0.1.6	
torch	      0.3.1	

・Packageのインストール方法
1 以下のコマンドを実行(pipを使用する場合)
pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
pip install http://download.pytorch.org/whl/torchvision-0.1.6-py3-none-any.whl

2 詳しくはを参照して下さい
https://pytorch.org/previous-versions/

・関連論文
Yoshina Takano, Wenwen Ouyangy, Suguru Ito, Tomohiro Harada and Ruck Thawonmas, "Applying Hybrid Reward Architecture to a Fighting Game AI," 2018 IEEE Conference on Computational Intelligence and Games (CIG 2018), Maastricht, The Netherlands, Aug.14-17. 2018.
http://www.ice.ci.ritsumei.ac.jp/~ruck/PAP/cig2018-takano.pdf 

・関連動画
https://youtu.be/9tPdGveY4CA
---------------------------------