# CodeCare
CodeCare: 초보 프로그래머를 위한 Gemma 4 튜터

## 프로젝트 요약

CodeCare는 프로그래밍 초보자를 위한 Gemma 4 기반 학습 우선 튜터입니다.

많은 초보자는 코딩을 배울 능력이 없어서 포기하는 것이 아닙니다. 에러 메시지가 이해되지 않고, 디버깅 과정이 막막하며, 정답 우선 AI 도구가 개념 이해 전에 수정 코드를 먼저 제공하기 때문에 학습 흐름이 끊기는 경우가 많습니다.

CodeCare는 프로그래밍 에러를 학습 기회로 바꿉니다. 과제 설명, 학생 코드, 에러 메시지를 입력하면 CodeCare는 다음과 같은 초보자 친화적 피드백을 생성합니다.

- 무슨 일이 발생했는지
- 왜 발생했는지
- 단계별 힌트
- 수정 방향
- 기억해야 할 개념
- 짧은 연습 문제

MVP는 Kaggle Notebook으로 구현되었으며, Kaggle Models에서 제공되는 Gemma 4 E4B IT를 사용합니다.

## 문제 정의

프로그래밍 초보자는 문법 자체보다 피드백 루프에서 더 자주 어려움을 겪습니다.

예를 들어 다음과 같은 에러를 보면:

```text
TypeError: can only concatenate str (not "int") to str
```

또는:

```text
NameError: name 'total_price' is not defined
```

초보자는 보통 다음을 알기 어렵습니다.

- 어느 줄을 봐야 하는지
- Python이 어떤 객체에 대해 불평하는지
- 어떤 개념이 관련되어 있는지
- 어떻게 질문해야 하는지
- 다음에 같은 실수를 피하려면 무엇을 기억해야 하는지

기존 AI 코딩 도구는 보통 빠르게 정답 코드를 제공하는 데 최적화되어 있습니다. 숙련된 개발자에게는 유용하지만, 초보자에게는 오히려 학습을 방해할 수 있습니다. 모델이 정답을 너무 빨리 주면 학생은 개념을 이해하기 전에 코드를 복사하게 됩니다.

CodeCare의 목표는 반대입니다. 정답 우선이 아니라 학습 우선입니다.

## 솔루션

CodeCare는 에러 메시지를 구조화된 학습 피드백으로 바꾸는 AI 프로그래밍 튜터입니다.

핵심 입력은 다음입니다.

- 과제 설명
- 학생 코드
- 에러 메시지

핵심 출력은 다음입니다.

1. 무슨 일이 발생했는가?
2. 왜 발생했는가?
3. Hint 1
4. Hint 2
5. 수정 방향
6. 기억해야 할 개념
7. 연습 문제

가장 중요한 설계 선택은 힌트 우선 튜터링입니다. CodeCare는 바로 정답으로 점프하지 않습니다. 먼저 학생이 에러를 읽고, 관련 개념을 연결하고, 다음에 무엇을 시도할지 생각하도록 돕습니다.

## 왜 Gemma 4인가?

Gemma 4는 CodeCare에 적합합니다. CodeCare는 단순 코드 완성 도구가 아니라 교육 도구이기 때문입니다.

Gemma 4를 사용한 이유는 다음과 같습니다.

- Gemma 4는 구조화된 자연어 설명을 생성할 수 있습니다.
- 섹션, 힌트, 개념 요약이 포함된 튜터링 프롬프트를 잘 따를 수 있습니다.
- 오픈 모델이므로 학교, 코딩 동아리, 부트캠프, 저비용 교육 환경에 적합합니다.
- 학생 코드가 폐쇄형 외부 API로 나가지 않는 로컬 또는 기관 통제형 배포로 확장할 수 있습니다.
- 프라이버시 친화적인 프로그래밍 교육을 현실적으로 만들 수 있습니다.

Notebook은 Kaggle Models에서 Gemma 4 E4B IT를 로드합니다.

```text
/kaggle/input/models/google/gemma-4/transformers/gemma-4-e4b-it/1
```

최종 Notebook 실행에서는 다음과 같은 모델 사용 증거를 기록합니다.

```text
generation_backend = gemma
gemma_ready = True
```

## 시스템 설계

CodeCare는 하나의 self-contained Kaggle Notebook으로 구현되었습니다.

Notebook에는 다음이 포함됩니다.

- 프로젝트 소개
- 초보자 에러 합성 데이터셋
- 프롬프트 설계
- Gemma 4 inference 설정
- 로컬 검토용 mock fallback
- CodeCare 튜터링 함수
- 데모 출력
- Generic Prompt vs CodeCare Prompt 평가
- 차트
- 한계점
- 향후 개선 방향
- `/kaggle/working` 아래로 export되는 결과물

주요 함수는 다음과 같습니다.

- `build_error_explainer_prompt(case)`
- `build_hint_generator_prompt(case)`
- `build_concept_explainer_prompt(concept)`
- `build_evaluator_prompt(case, generic_answer, codecare_answer)`
- `generate_with_gemma(prompt, max_new_tokens=512)`
- `mock_generate(prompt)`
- `explain_error(case)`
- `generate_hints(case)`
- `explain_concept(concept)`
- `run_codecare_pipeline(case)`
- `evaluate_answers(case, generic_answer, codecare_answer)`
- `run_evaluation(cases)`

## 프롬프트 설계

CodeCare의 핵심 프롬프트는 Gemma 4가 초보자 친화적인 프로그래밍 튜터처럼 행동하도록 지시합니다.

```text
You are CodeCare, an AI programming tutor for beginner students.

Your goal is not to immediately give the final answer.
Your goal is to help the student understand the error and learn how to fix it.
```

프롬프트 규칙은 다음과 같습니다.

1. 에러를 초보자가 이해할 수 있는 언어로 설명합니다.
2. 가능한 원인을 식별합니다.
3. 최종 수정 전에 단계별 힌트를 제공합니다.
4. 과도하게 어려운 용어를 피합니다.
5. 수정 코드를 제공하는 경우 왜 작동하는지 설명합니다.
6. 정보가 부족하면 필요한 맥락을 요청합니다.
7. 입력에 없는 내용을 아는 척하지 않습니다.

이 프롬프트 설계가 CodeCare와 일반 코딩 assistant의 핵심 차이입니다.

## 데이터셋

Notebook은 초보자 Python 에러 데이터셋을 코드 내부에서 직접 생성합니다.

데이터셋은 24개의 케이스를 포함하며 다음 범주를 다룹니다.

- SyntaxError
- NameError
- TypeError
- IndexError
- KeyError
- ValueError
- IndentationError
- ModuleNotFoundError
- AttributeError
- 함수 인자 오류
- 반복문 실수
- 조건문 실수
- 리스트 실수
- 딕셔너리 실수
- 문자열/숫자 변환 실수
- pandas 컬럼명 오류

각 케이스에는 다음 필드가 있습니다.

- `case_id`
- `category`
- `assignment`
- `student_code`
- `error_message`
- `expected_concept`
- `reference_fix`

이 데이터셋은 완전한 벤치마크가 아니라, 초보자가 자주 겪는 오류 유형에 대해 튜터가 어떻게 반응하는지 보여주는 MVP 평가셋입니다.

## MVP 기능

### 1. 에러 설명기

에러 설명기는 과제 설명, 학생 코드, 에러 메시지를 입력받아 무슨 일이 발생했는지와 왜 발생했는지를 설명합니다.

어려운 용어를 과도하게 사용하지 않고, 초보자가 가장 먼저 확인해야 할 부분에 집중합니다.

### 2. 단계별 힌트 생성기

힌트 생성기는 다음과 같은 hint ladder를 만듭니다.

- Hint 1: 어디를 봐야 하는지
- Hint 2: 관련 개념
- Hint 3: 어떻게 수정할 수 있는지
- Optional fix: 힌트 이후에만 제공

학생이 바로 정답을 보기 전에 스스로 생각할 기회를 제공하기 때문에 능동 학습을 지원합니다.

### 3. 개념 설명기

개념 설명기는 다음과 같은 프로그래밍 개념을 쉽게 설명합니다.

- 타입 변환
- 리스트 인덱싱
- 딕셔너리 키
- 함수 인자
- 반복문
- 조건문
- pandas 컬럼명

각 설명에는 다음이 포함됩니다.

- 쉬운 설명
- 간단한 예시 코드
- 흔한 실수
- 기억할 핵심 문장
- 짧은 연습 문제

### 4. 답변 품질 평가기

평가기에서는 다음 두 답변을 비교합니다.

- 일반적인 answer-first 응답
- CodeCare의 learning-first 응답

평가 기준은 다음입니다.

- Beginner-friendly
- Hint-first
- Concept clarity
- Technical correctness
- Actionability
- Active learning

각 항목은 1점에서 5점으로 평가됩니다.

## 평가

Notebook은 24개 합성 케이스 전체에 대해 평가를 수행합니다.

목표는 CodeCare가 모든 버그를 완벽히 고친다고 주장하는 것이 아닙니다. 목표는 초보자에게 중요한 교육적 응답 품질을 측정 가능한 기준으로 보여주는 것입니다.

평가 비교 대상은 다음입니다.

- Generic Prompt: 에러를 바로 고치라고 요청하는 프롬프트
- CodeCare Prompt: 설명, 힌트, 개념 연결, 연습 문제를 요청하는 프롬프트

Notebook은 다음 결과물을 생성합니다.

- pandas 평가표
- 요약표
- 전체 평균 점수표
- Generic Prompt vs CodeCare Prompt를 비교하는 matplotlib bar chart

## 결과

평가 결과 CodeCare는 일반적인 answer-first prompt보다 초보자 학습 목표에 더 잘 맞는 응답을 생성합니다.

핵심 개선은 단순 점수 차이가 아니라 행동 차이입니다.

- Generic Prompt는 수정 코드로 빠르게 이동하는 경향이 있습니다.
- CodeCare Prompt는 에러를 설명하고, 힌트를 제공하고, 개념을 식별하고, 연습 문제를 제안합니다.

이러한 구조는 교실이나 자기주도 학습 환경에 더 적합합니다.

## 생성된 Artifacts

Notebook은 `/kaggle/working` 아래에 주요 결과물을 export합니다.

```text
codecare_dataset_cases.csv
codecare_demo_outputs.csv
codecare_evaluation_results.csv
codecare_evaluation_summary.csv
codecare_overall_scores.csv
codecare_submission_checklist.csv
codecare_model_usage_proof.csv
codecare_submission_report.md
```

가장 중요한 artifact는 다음입니다.

```text
codecare_model_usage_proof.csv
```

이 파일은 최종 Notebook 실행이 실제 Gemma backend를 사용했는지 기록합니다.

```text
generation_backend = gemma
gemma_ready = True
```

## Writeup에 사용할 이미지 제안

이미지를 하나만 사용할 수 있다면 평가 그래프를 추천합니다.

- Generic Prompt vs CodeCare Prompt 평균 점수 그래프

여러 이미지를 사용할 수 있다면 다음을 포함하는 것이 좋습니다.

1. Live demo output 스크린샷
2. 평가 점수 그래프
3. `generation_backend = gemma`가 보이는 model usage proof 표
4. submission checklist 표

이 이미지들은 제품 동작, 기술적 증거, 평가 스토리를 함께 보여줍니다.

## 기대 효과

CodeCare는 초보자가 프로그래밍을 포기하게 만드는 중요한 문제를 다룹니다. 에러 메시지가 이해되지 않고, 다음 행동을 알 수 없어서 학습이 중단되는 문제입니다.

활용 가능 영역은 다음과 같습니다.

- 입문 프로그래밍 수업
- 코딩 부트캠프
- 방과 후 코딩 동아리
- 자기주도 학습
- LMS 연동 피드백
- 프라이버시 친화적인 교실 배포

Gemma 4가 오픈 모델이기 때문에 CodeCare는 비용과 개인정보 보호가 중요한 환경에서도 활용 가능성이 있습니다. 이는 고가의 폐쇄형 API를 사용하기 어려운 학교와 학습 커뮤니티에 특히 중요합니다.

## 한계점

CodeCare는 MVP이며 완벽한 디버거가 아닙니다.

현재 한계는 다음과 같습니다.

- 실행 맥락이 부족하면 원인을 잘못 추론할 수 있습니다.
- 일부 버그는 패키지 버전, hidden test, 파일, runtime state가 필요합니다.
- 데이터셋은 합성 데이터이며 실제 학생 디버깅 사례로 확장되어야 합니다.
- 평가 rubric은 프로토타입에는 유용하지만 교사와 학습자를 통한 검증이 필요합니다.
- 현재 구현은 Notebook prototype이며 production web application은 아닙니다.
- 실제 교실 적용 전에는 교사 검토와 안전성 점검이 필요합니다.

## 향후 개선 방향

향후 개선 방향은 다음과 같습니다.

- 한국어/영어 이중 언어 튜터링
- LMS 연동
- 교사용 dashboard
- 학생 학습 리포트
- 난이도별 힌트 조절
- 오프라인 classroom mode
- 실제 학생 피드백 수집
- 더 큰 평가 데이터셋
- 추가 프로그래밍 언어 지원

## 결론

CodeCare는 Gemma 4를 활용해 프로그래밍 에러를 학습 기회로 바꾸는 프로젝트입니다.

이 프로젝트는 의도적으로 learning-first 접근을 취합니다. 단순히 수정 코드를 제공하는 것이 아니라, 초보자가 무슨 일이 발생했는지, 왜 발생했는지, 어떤 개념이 관련되어 있는지, 다음에 무엇을 시도해야 하는지 이해하도록 돕습니다.

Kaggle Notebook은 데이터셋, 프롬프트, Gemma 4 inference, 데모, 평가, 차트, export artifacts를 포함한 완전한 MVP를 보여줍니다.

CodeCare는 Gemma 4가 접근 가능하고 저비용이며 프라이버시 친화적인 프로그래밍 교육을 지원할 수 있음을 보여줍니다.

