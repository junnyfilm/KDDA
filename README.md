# Knowledge Distillation with Domain Adaptation


![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cbbebd30-e681-47e0-9048-75c6faebc7af/01c5b61c-ba4a-4aa1-bd5f-869c1eedfe2a/Untitled.png)
- 기계 설비에서 가장 많이 사용 되는 시스템인 모터의 고장진단
- 모터의 고장진단을 위한 센서는 주로 진동센서와 전류센서가 있음
- 진동센서의 경우 성능이 우수한 편이지만, 센서 설치의 불편함, 센서 설치 위치에 따른 신호 일반화 불가능, 외부의 물리적 노이즈에 취약함 등의 단점이 있기 때문에 현장에서 사용하기 어려움
- 따라서 전류센서를 현장에서 사용하는 것이 유리하지만, 진동신호를 활용하여 성능을 개선하는 연구가 필요함
  
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cbbebd30-e681-47e0-9048-75c6faebc7af/ef9f4b58-dd60-41e6-af1b-006c9d070325/Untitled.png)
- 이를 위한 방법으로 지식증류를 활용할 수 있음.
- 일반적으로 Teacher model 은 large model, Student model은 small model 로 경량화를 위한 목적으로 많이 사용되지만, 본 연구에서는 Teacher model 을 전류와 진동의 멀티모달 모델, Student model 을 전류 단일 모델로 두어 학습을 진행
- 학습 단계에서 지식증류를 통해 전류 기반의 모델 성능을 개선하고 현장에서도 성능을 우수하게 유지할 수 있기 위함
  
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cbbebd30-e681-47e0-9048-75c6faebc7af/553b6548-9d3c-4e79-a3ad-52a8d9ba8889/Untitled.png)
Teacher model train

1) SAKD: spatial attention 기반으로 각 모달의 중요부 집중하여 학습 및 self distillation으로 전류 피쳐성능 개선

2) VGCA: Cross attention으로 피쳐 퓨전하여 mixed 피쳐 성능 개선 (전류 성질 갖는 진동 피쳐가 됨)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0c7497be-6dab-4071-ab20-129afbabc95d/Untitled.png)
Student model train

3)DAM-CMKD : 어텐션으로 만들어진 피쳐부에 대해서 지식증류 진행

Knowledge distillation loss와  Classification loss 사이의 가중치 계산 방법, 에포크가 늘어남에 따라 분류 손실 함수에 가중치를 늘려주는 방향으로 진행


- 이를 현장에 적용하기 위하여 Domain Adaptation 과 결합
- 기존 DANN의 알고리즘 한계를 극복하기위하여 Multi-source Domain Adaptation을 제안하였고, 이에 각 소스별 영향을 미치는 정도를 조절하기 위하여 Weight를 CMD로 계산
- 제안 방법 CMD based Weighted Multi-source Domain Adaptation
  
![image](https://github.com/junnyfilm/KDDA/assets/109502364/bb8aab60-c69d-47ac-b3ab-85c8caa7a92f)
융합모델 제안
