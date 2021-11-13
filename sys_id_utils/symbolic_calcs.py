import sympy

def transfer_function_lpi():
    print() 
    I, d, gp, gi, c,  s = sympy.symbols('I d gp gi c s')
    
    A = sympy.Matrix([[-(d + gp)/I, 0], [-1, 0]])
    B = sympy.Matrix([[gp/I], [1]])
    C = sympy.Matrix([[1, 0]])
    Id = sympy.Matrix([[1,0], [0,1]])
    Phi = (s*Id - A).inv()
    H = (C*Phi*B)[0,0]
    H = sympy.simplify(H)
    print('P Controller')
    print('-----------------')
    print(H)
    print()
    
    A = sympy.Matrix([[-(d + gp)/I, gi/I], [-1, 0]])
    B = sympy.Matrix([[gp/I], [1]])
    C = sympy.Matrix([[1, 0]])
    Id = sympy.Matrix([[1,0], [0,1]])
    Phi = (s*Id - A).inv()
    H = (C*Phi*B)[0,0]
    H = sympy.simplify(H)
    num, den = sympy.fraction(H)
    den = sympy.collect(den,'s')
    print('PI Controller')
    print('-----------------')
    print(num/den)
    print()
    
    A = sympy.Matrix([[-(d + gp)/I, gi/I], [-1, -c]])
    B = sympy.Matrix([[gp/I], [1]])
    C = sympy.Matrix([[1, 0]])
    Id = sympy.Matrix([[1,0], [0,1]])
    Phi = (s*Id - A).inv()
    H = (C*Phi*B)[0,0]
    H = sympy.simplify(H)
    num, den = sympy.fraction(H)
    num = sympy.expand(num)
    num = sympy.collect(num,s)
    den = sympy.collect(den,s)
    print('LPI Controller')
    print('-----------------')
    print(num/den)

def transfer_function_2x2():
    print()
    a11, a12, a21, a22, b1, b2, s = sympy.symbols('a11 a12 a21 a22 b1 b2 s')
    A = sympy.Matrix([[a11, a12], [a21, a22]])
    B = sympy.Matrix([[b1], [b2]])
    C = sympy.Matrix([[1, 0]])
    Id = sympy.Matrix([[1,0], [0,1]])
    Phi = (s*Id - A).inv()
    H = (C*Phi*B)[0,0]
    H = sympy.simplify(H)
    num, den = sympy.fraction(H)
    num = sympy.expand(num)
    num = sympy.collect(num,s)
    den = sympy.collect(den,'s')
    print('General 2D system')
    print('-----------------')
    print(num/den)
