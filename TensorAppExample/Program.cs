using System;
using System.Numerics.Tensors;

 float Nonlin(float x, bool deriv = false)
    {
        if (deriv)
        {
            return x * (1 - x);
        }
        return (float)(1 / (1 + Math.Exp(-x)));
    }

Random random = new Random(1);

var syn0 = Tensor.Create(new float[] {
        (float) random.NextDouble() * 2 - 1,
        (float) random.NextDouble() * 2 - 1,
        (float) random.NextDouble() * 2 - 1 }, [1, 3]);

void Train(int epochs) 
    {

    // Массив выходных значений
    var x = Tensor.Create(new float[] { 
            0, 0, 1,
            0, 1, 1,
            1, 0, 1,
            1, 1, 1 }, [4, 3]);

    // Массив выходных значений
    var y = Tensor.Create(new float[] { 
            0, 1, 1, 1 }, [1, 4]);

    // Тренировка
    for (int i = 0; i < epochs; i+=1) {
        var l0 = x;
        var l1 = Tensor.Multiply(l0, syn0);

        // Костыль для получения скалярного произведения при помощи встроенных функций
        var tl1 = Tensor.Create(new float[] { 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, [4, 3]); var l11 = Tensor.Multiply(l1, tl1);
        var tl2 = Tensor.Create(new float[] { 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0 }, [4, 3]); var l12 = Tensor.Multiply(l1, tl2);
        var tl3 = Tensor.Create(new float[] { 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0 }, [4, 3]); var l13 = Tensor.Multiply(l1, tl3);
        var tl4 = Tensor.Create(new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1 }, [4, 3]); var l14 = Tensor.Multiply(l1, tl4);

        // Скалярное произведение матриц, нормализованное
        var l1dot = Tensor.Create(new float[] { 
            Nonlin(Tensor.Sum(l11)),
            Nonlin(Tensor.Sum(l12)),
            Nonlin(Tensor.Sum(l13)),
            Nonlin(Tensor.Sum(l14)) }, [1, 4]);

        // Получим значение ошибки
        var l1_error = Tensor.Subtract(y, l1dot);
        
        // Получим производные
        var l1_deriv = Tensor.Create(new float[] {
            Nonlin(l1dot.ElementAt(0), true),
            Nonlin(l1dot.ElementAt(1), true),
            Nonlin(l1dot.ElementAt(2), true),
            Nonlin(l1dot.ElementAt(3), true)
        }, [1, 4]);

        // Получим значение отклонения
        var l1_delta = Tensor.Multiply(l1_error, l1_deriv);

        // Обновим веса
        var l0_t = Tensor.Transpose(l0);
        var l0_t_m = Tensor.Multiply(l0_t, l1_delta);

        var l0_t_m1 = Tensor.Create(new float[] { 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 }, [3, 4]);
        var l0_t_m11 = Tensor.Multiply(l0_t_m, l0_t_m1);
        var l0_t_m2 = Tensor.Create(new float[] { 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0 }, [3, 4]);
        var l0_t_m12 = Tensor.Multiply(l0_t_m, l0_t_m2);
        var l0_t_m3 = Tensor.Create(new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 }, [3, 4]);
        var l0_t_m13 = Tensor.Multiply(l0_t_m, l0_t_m3);

        var l0_t_m_sum = Tensor.Create(new float[] {
            Tensor.Sum(l0_t_m11),
            Tensor.Sum(l0_t_m12),
            Tensor.Sum(l0_t_m13)}, [1, 3]);

        syn0 = Tensor.Add(syn0, l0_t_m_sum);
    }
}

void Eval()
{
    var z = Tensor.Create(new float[] {
            0, 0, 1,
            0, 1, 1,
            1, 0, 1,
            1, 1, 1 }, [4, 3]);

    Console.WriteLine("Syn0 weights: ");
    Console.WriteLine(syn0.ElementAt(0) + ", " + syn0.ElementAt(1) + ", " + syn0.ElementAt(2));

    var l1 = Tensor.Multiply(z, syn0);

    var tl1 = Tensor.Create(new float[] { 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, [4, 3]); var l11 = Tensor.Multiply(l1, tl1);
    var tl2 = Tensor.Create(new float[] { 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0 }, [4, 3]); var l12 = Tensor.Multiply(l1, tl2);
    var tl3 = Tensor.Create(new float[] { 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0 }, [4, 3]); var l13 = Tensor.Multiply(l1, tl3);
    var tl4 = Tensor.Create(new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1 }, [4, 3]); var l14 = Tensor.Multiply(l1, tl4);

    var l1dot = Tensor.Create(new float[] {
            Nonlin(Tensor.Sum(l11)),
            Nonlin(Tensor.Sum(l12)),
            Nonlin(Tensor.Sum(l13)),
            Nonlin(Tensor.Sum(l14)) }, [1, 4]);

    Console.WriteLine("Outputs nonlin:");
    Console.WriteLine(l1dot.ElementAt(0) + ", " + l1dot.ElementAt(1) + ", " + l1dot.ElementAt(2) + ", " + l1dot.ElementAt(3));
}

void Print_Init()
{
    Console.WriteLine("Init values:");
    Console.WriteLine(syn0.ElementAt(0) + ", " + syn0.ElementAt(1) + ", " + syn0.ElementAt(2));
}

Print_Init();
Train(150);
Eval();