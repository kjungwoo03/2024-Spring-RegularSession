# CNN Assignment Leaderboard

상위 10명의 Accuracy, Hyperparameter는 아래와 같습니다!

| Rank | Name  | Accuracy | Epoch | Learning Rate | 딥러닝스터디 | 칭호 |
|:----:|:-----:|:--------:|:-----:|:-------------:|:--------:|:---:|
| 1 | 👑 김현진 👑 |  99.60% | 40 | 0.0001 | 1팀 | 🛡️ 정복자 🏹 |
| 2 | 🥈 박태정 🥈 |  99.33% | 40 | 0.0001 | 4팀 | ⚔️ 찬탈자 ⚔️ |
| 3 | 🥉 신재우 🥉 |  99.17% | 40 | 0.0001 | 3팀 | 어글리 코리안 |
| 4 | 김정우 |  99.17% | 120 | 0.001 | 3팀 | - |
| 😈 | 윤형진 |  99.00% | 100 | 0.0001 | - | 📚 출제자 📝 |
| 5 | 남예진 |  98.83% | 40 | 0.001  | 3팀 | 차기 학술부장 |
| 6 | 임창재 |  98.00% | 40 | 0.0001 | 1팀 | - |
| 7 | 송예원 |  97.83% | 40 | 0.0001 | 3팀 | - |
| 8 | 김주연 |  96.83% | 40 | 0.001  | 4팀 | - |
| 9 | 박성원 |  96.77% | 30 | 0.001  | 2팀 | - |
| 10 | 김여원 |  96.33% | 40 | 0.001  | 5팀 | - |


# CNN 과제 해설

안녕하세요, 10기 학술부장 윤형진입니다. `PyTorch`에 능숙한 우리 운영진도 많은 시간이 걸렸던 과제인 만큼, 어렵고 긴 과제를 수행하시느라 고생많으셨습니다. 딥러닝 프레임워크가 처음인 분들이 많았음에도 성실하게 과제를 해주셔서, **여러분께서 제출한 과제를 읽어보며 DSL의 미래가 밝다**는 점을 새삼 느겼습니다.

어렵게 한 과제인만큼, 본 해설을 읽어보시면서 과제를 하면서는 보지 못하셨을 수도 있는 점들을 생각해보셨으면 좋겠습니다.

모범 답안 발췌 : `11기 김현진`, `11기 남예진`, `10기 유현동`

### 문제 1.

Q. 모델의 학습을 train, validation, test로 나눠서 진행하는 이유는 무엇인가요?

- (힌트 : 굳이 validation을 끼워넣는 이유는 무엇인가요?)

A. 모범답안 : `11기 김현진`

- 모델의 학습을 train, validation, test로 나눠 진행하는 이유는 모델의 성능을 평가하는 동시에 일반화(generalization) 성능을 확인하기 위함입니다.
- Train dataset은 모델의 파라미터를 학습하는 데 사용되며, 훈련 데이터를 사용한 학습을 통해 모델은 최적의 파라미터를 찾아갑니다.
- Validation data는 학습 중 모델의 성능을 모니터링하고 이를 통해 과적합 여부를 확인하기 위한 data입니다.
- Test dataset을 통해 모델의 일반화 성능을 최종적으로 평가하며, 모델이 이전에 학습하지 못한 데이터에 대해 얼마나 잘 예측하는지를 알 수 있습니다.
- 위에서 Overfitting을 방지하기 위해 train dataset에는 포함되지 않은 데이터들로 validation/test dataset을 구성한다고 했습니다. 그렇다면 overfiitting 되지 않고 잘 학습되고 있는 모델은 어떤 모습이고, 잘못된 길 (overffitting) 을 걷고 있는 모델은 어떤 모습일까요?
- Overfit 모델의 경우, ‘**Train loss는 지속적으로 감소하나, 일정 epoch에서부터 Validation loss가 Train loss의 감소추세를 따라가지 못하고 수렴하는**’ 손실함수 곡선을 그리게됩니다. 즉, Validation loss가 Train loss와 함께 잘 감소하고 있다면, ‘일반적’으로 적용시킬 수 있는 feature들을 잘 학습하고 있다고 평가할 수 있습니다.
- 따라서 매 Epoch 마다 계산되는 Validation loss의 감소추이를 보고 학습의 중단/진행 여부, 하이퍼파라미터 조정 등의 동작을 취하게 됩니다.

### 문제 2.

Q. 모델의 generalizability를 높여주기 위해 Augmentation을 사용할 수 있다고 했습니다.

- PyTorch 구현상으로는 Train, Validation, Test 모든 단계에 Augmention를 구현할 수 있습니다만, 실제로는 특정 단계에서만 Augmention를 수행하게 됩니다. 어느 단계에서 Augmentation이 적용되어야하는지와, 그 이유에 대해서 설명해주세요.

**정정공지** : 해당 과제에서 적용된 것은 RandAugment가 아닌 일반적인 Augmentation입니다. RandAugment는 적용될 Transformation의 수/변형정도 모두 확률에 기반한 데이터 증강기법으로, 해당 과제처럼 고정된 수만의 Transformation을 적용하는 것과는 조금 다릅니다.

A. Train dataset에’만’ 적용되어야합니다.

- 문제1에서 보았듯이, Validation dataset은 학습과정을 실시간으로 모니터링하기 위해 Train dataset에는 없는 데이터들로 구성되어 있습니다. 즉, Validation dataset 또한 Test Dataset처럼 모델이 Train datatset에선 본 적 없는 데이터를 이용해야만 모델의 학습상황을 올바르게 모니터링할 수 있습니다.
- Transformation은 학습 데이터가 부족한 상황에서 Generalizability 향상을 위해 취하는 학습상의 전략이므로, 어떤 방식으로든 모델을 평가하는 단계에서는 ‘변형되지 않은’ 데이터로만 구성하여야합니다.

### 문제 3.

Q. `CustomImageDataset`과 `DataLoader` 클래스의 차이점에 대해서 설명해주세요.

- (힌트 : 두 클래스의 리턴 형식에 어떤 차이가 있나요? 어디서부터 Batch 단위로 리턴되나요?)

A. 모범답안 : `10기 유현동`

- `CustomImageDataset` 클래스는 주로 이미지 파일을 읽고 전처리하여 모델에 입력으로 공급할 수 있는 형태로 데이터를 제공합니다.각 샘플은 이미지와 해당 이미지의 라벨로 구성됩니다.
- `DataLoader` 클래스는 데이터셋을 미니배치로 나누고 셔플링 및 병렬로 데이터를 로드하는 데 사용됩니다.
- 따라서 `CustomImageDataset` 클래스는 데이터셋을 정의하고 데이터셋에 대한 개별 샘플을 가져오는 데 사용되고, `DataLoader` 클래스는 데이터셋을 미니배치 단위로 로드하고 셔플링하여 모델에 제공하는 데 사용됩니다.

### 문제 4.

Q. 주어진 과제는 Binary Classification 태스크입니다.

- 이중 분류를 위해서는 손실함수로 Binary Cross Entropy를 사용한다고 세션에서 배웠습니다.
- PyTorch에는 Binary Cross Entropy를 학습에 사용할 수 있는 2가지 방법이 있는데요, `nn.BCELoss`와 `nn.BCEWithLogitsLoss`입니다.
- 이 둘은 같아 보이지만 구현상에서 명백한 차이점이 있습니다. 이 둘의 차이점에 대해서 서술해주세요.

A. 모범답안 : `11기 남예진`

- `nn.BCELoss` : 이진분류를 위한 binary cross entropy 손실함수입니다. 이 함수를 사용할 때는 모델의 출력이 0과 1 사이의 확률값이 되도록 해야 합니다. 즉, 모델의 마지막 레이어에서 sigmoid 활성화 함수를 적용해야 합니다. (Sigmoid를 씌우지 않은 모델의 출력값은 Logit값이기 때문입니다)
- `nn.BCEWithLogitsLoss` : `nn.BCELoss`와 기본적으로 동일한 이진분류 binary cross entropy 손실함수이지만, 시그모이드 함수를 내부적으로 적용합니다. 따라서 모델의 출력에서 시그모이드 함수를 별도로 적용할 필요가 없습니다. `nn.BCEwithLogitsLoss`는 로짓과 함께 시그모이드 함수를 계산함으로써 수치적 안정성을 높일 수 있습니다.

### 문제 5.

Q. 학습이 끝난 이후 `val_loss`, `train_loss` 변화 추이 그래프와 `val_acc`, `train_acc` 변화 추이 그래프를 첨부해주세요.

A.

- Accuracy : Validation accuracy가 Train accuracy에 비해 조금 ‘bumpy’하지만, 학습이 진행됨에 따라 안정적으로 상승함을 볼 수 있습니다.
    - <img width="1364" alt="스크린샷 2024-02-20 오후 7 22 18" src="https://github.com/DataScience-Lab-Yonsei/2024-Spring-RegularSession/assets/98018819/8453649a-2be6-4ff7-9436-8c742d3a6cf4">

        
- Loss : Validation loss가 Train loss에 비해 조금 ‘bumpy’하지만, 학습이 진행됨에 따라 Train Loss와 비슷한 수준으로, 안정적인 수렴을 하고있음을 볼 수 있습니다.
    - <img width="1364" alt="스크린샷 2024-02-20 오후 7 22 49" src="https://github.com/DataScience-Lab-Yonsei/2024-Spring-RegularSession/assets/98018819/c72fcc76-b55f-4041-9b1c-ae0b1505924c">

        

### 문제 7.

Q. 전달받으신 파일들 중, Run_NoAugment.ipynb라는 파일이 있을 겁니다. 지금 보고 계신 노트북과 똑같이 학습을 시키시되, 이번에는 학습 시 data augmentation을 적용시키지 않고 진행하십시오.

- 이후 augmentation을 적용시키지 않은 모델의 train loss와 val loss의 차트를 아래에 붙여놓으시고, augmentation을 적용한 모델의 학습 양상과 어떤 점이 다른 지와 그 이유도 함께 설명해주세요.

A. W/ Augmentation (진한 실선) VS No Augmentation (흐린 점선)

- <img width="1364" alt="스크린샷 2024-02-20 오후 7 23 15" src="https://github.com/DataScience-Lab-Yonsei/2024-Spring-RegularSession/assets/98018819/06f19469-19c5-4679-98dc-7282d8d9dea4">

    - 함께 비슷한 수준으로 수렴해나가는 Augmentation이 적용된 모델과는 달리, Augmentation이 적용되지 않은 모델은 급격히 줄어든 Train loss에 비해 Validation loss는 수렴하지 않는, 과적합된 모습을 볼 수 있습니다.
    - 간혹 Augmentation을 적용시키지 않은 경우가 더 작은 Loss를 보인 경우가 있는데, 이 경우 학습을 더 길게 진행시켰을 경우 Augmentation을 적용시킨 모델이 더 작은 Loss를 가지게 될 여지가 있습니다. Augmentation을 적용시키지 않은 모델의 경우, 매 Epoch 마다, 변형되지 않은, 동일한 데이터를 보기 때문에 연산량이 적고 최적화 난이도가 작은 Epoch에서는 비교적 쉬울 수 있기 때문입니다.
    - 이와 같이 Validation Dataset을 이용하면 실시간으로 모델의 학습상황을 모니터링하고, 학습 진행/중단 여부 및 하이퍼 파라미터 튜닝 등의 판단을 내릴 수 있습니다.
