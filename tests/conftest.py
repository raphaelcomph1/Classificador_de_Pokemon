import pytest

# Testes de partes que o aluno implementa (TODO). Enquanto a função levanta
# NotImplementedError, o teste aparece como "xfail". Depois de implementado, ele passa
# a valer de verdade: uma asserção errada vira FALHA normal.
todo = pytest.mark.xfail(raises=NotImplementedError, reason="TODO do aluno", strict=False)
