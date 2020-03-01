import Model
from Sundail import Sundail 
import numpy as np
from Model import TDormandPrince
X=np.array([
    -2.566123740124270e+7,
    1.339350231544666e+8,
    5.805149372446711e+7,
    -29.83549561177192,
    -4.846747552523134,
    -2.100585886567924
])

# Y=np.zeros(6)
# for i in range(6):
#     Y[i]=X[i]

E_S=Sundail(0,50000,1,X)

Integrator=TDormandPrince()

Integrator.setPrecision(1e-16)

Integrator.Run(E_S)

Result=E_S.getResult()

print("hello")