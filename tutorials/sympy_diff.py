import sympy

s = sympy.symbols("s")
exp1 = 1 * s**2 + 0 * s + 0
exp2 = 0 * s**2 + 1 * s + 0
exp3 = 0 * s**2 + 0 * s + 1


temp = sympy.sqrt(exp1.diff(s) ** 2 + exp2.diff(s) ** 2 + exp3.diff(s) ** 2)
print(temp)
print(sympy.integrate(temp, (s, 0, 1)).evalf())


from weldx import MathematicalExpression

params = dict(a=[1, 0, 0], b=[0, 1, 0], c=[0, 0, 1])
me = MathematicalExpression("a * s**2 + b * s + c", parameters=params)

der_sq = []
for i in range(3):
    ex = me.expression
    subs = [(k, v[i]) for k, v in me.parameters.items()]

    der_sq.append(ex.subs(subs).diff("s") ** 2)
print(der_sq)
expr_l = sympy.sqrt(der_sq[0] + der_sq[1] + der_sq[2])
print(expr_l)
print(sympy.integrate(expr_l, ("s", 0, 1)))
