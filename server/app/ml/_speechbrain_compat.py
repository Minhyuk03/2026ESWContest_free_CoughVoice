"""speechbrain 지연 모듈이 무관한 코드에서 깨어나 죽는 것을 막는다.

speechbrain은 선택적 의존성(k2 등)을 `LazyModule`로 sys.modules에 올려두고,
속성에 처음 접근할 때 진짜 import를 수행한다. 문제는 **모듈을 훑는 무관한 코드**가
그 접근을 유발한다는 점이다. 대표적으로 `linecache.getlines()`가
`getattr(mod, "__file__", None)`을 호출하는데, 이때 LazyModule이 깨어나
미설치 의존성을 끌어오다 ImportError로 죽는다.

linecache는 **예외 traceback을 출력할 때** 호출되므로, 결과적으로 원래 예외가
이 ImportError에 가려져 무엇이 실패했는지 알 수 없게 된다(2026-08-25 맥미니에서
POST /events가 500을 냈는데 로그에 k2 ImportError만 남아 원인 추적이 막혔다).

해결: LazyModule의 `__dict__`에 `__file__`을 직접 넣어 둔다. 그러면 일반 속성
조회로 값을 찾으므로 `__getattr__`(지연 import)이 호출되지 않는다. 가짜 경로라
linecache는 소스를 못 찾고 조용히 건너뛴다 — 우리가 원하는 동작이다.

앞서 겪은 같은 계열 증상 두 건은 import 순서를 고정해 우회했으나
(cough_gate의 matplotlib, 학습 스크립트의 torch._dynamo), 예외 출력 경로에서는
순서로 막을 수 없어 이 방식이 필요하다.
"""
from __future__ import annotations

import sys


def neutralize_lazy_modules() -> int:
    """speechbrain 지연 모듈에 __file__을 심어 두고, 손댄 개수를 반환한다.

    speechbrain을 import한 뒤에 호출해야 한다. 그 전에는 대상이 sys.modules에 없다.
    """
    patched = 0
    for name, mod in list(sys.modules.items()):
        if not name.startswith("speechbrain") or mod is None:
            continue
        try:
            if "__file__" not in vars(mod):
                mod.__file__ = f"<lazy:{name}>"
                patched += 1
        except (AttributeError, TypeError):
            pass  # 손댈 수 없는 객체는 건너뛴다 — 최선 노력으로 충분하다
    return patched
