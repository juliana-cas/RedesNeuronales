package cue.edu;

import java.util.Random;

public class SimpleNeuralNetwork {
    private static final int INPUT_NEURONS = 2;      // Neuronas de entrada
    private static final int HIDDEN_NEURONS = 4;     // Neuronas en la capa oculta
    private static final int OUTPUT_NEURONS = 1;     // Neurona de salida
    private static final double LEARNING_RATE = 0.1; // Tasa de aprendizaje

    private double[] weightsInputHidden;             // Pesos de entrada a oculta
    private double[] weightsHiddenOutput;            // Pesos de oculta a salida
    private Random random;

    public SimpleNeuralNetwork() {
        random = new Random();
        weightsInputHidden = new double[INPUT_NEURONS * HIDDEN_NEURONS];
        weightsHiddenOutput = new double[HIDDEN_NEURONS * OUTPUT_NEURONS];

        // Inicializamos los pesos con valores aleatorios
        for (int i = 0; i < weightsInputHidden.length; i++) {
            weightsInputHidden[i] = random.nextDouble() - 0.5;
        }
        for (int i = 0; i < weightsHiddenOutput.length; i++) {
            weightsHiddenOutput[i] = random.nextDouble() - 0.5;
        }
    }

    // Función de activación sigmoide
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // Derivada de la función sigmoide
    private double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }

    // Propagación hacia adelante
    public double forward(double[] inputs) {
        // Capa oculta
        double[] hiddenLayerOutputs = new double[HIDDEN_NEURONS];
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            double sum = 0;
            for (int j = 0; j < INPUT_NEURONS; j++) {
                sum += inputs[j] * weightsInputHidden[j + i * INPUT_NEURONS];
            }
            hiddenLayerOutputs[i] = sigmoid(sum);
        }

        // Capa de salida
        double output = 0;
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            output += hiddenLayerOutputs[i] * weightsHiddenOutput[i];
        }
        return sigmoid(output);
    }

    // Retropropagación
    public void train(double[] inputs, double target) {
        // Propagación hacia adelante
        double[] hiddenLayerOutputs = new double[HIDDEN_NEURONS];
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            double sum = 0;
            for (int j = 0; j < INPUT_NEURONS; j++) {
                sum += inputs[j] * weightsInputHidden[j + i * INPUT_NEURONS];
            }
            hiddenLayerOutputs[i] = sigmoid(sum);
        }

        double output = 0;
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            output += hiddenLayerOutputs[i] * weightsHiddenOutput[i];
        }
        output = sigmoid(output);

        // Cálculo de errores
        double outputError = target - output;
        double[] hiddenErrors = new double[HIDDEN_NEURONS];

        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            hiddenErrors[i] = outputError * weightsHiddenOutput[i] * sigmoidDerivative(hiddenLayerOutputs[i]);
        }

        // Actualización de pesos de oculta a salida
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            weightsHiddenOutput[i] += LEARNING_RATE * outputError * sigmoidDerivative(output) * hiddenLayerOutputs[i];
        }

        // Actualización de pesos de entrada a oculta
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < INPUT_NEURONS; j++) {
                weightsInputHidden[j + i * INPUT_NEURONS] += LEARNING_RATE * hiddenErrors[i] * inputs[j];
            }
        }
    }

    public static void main(String[] args) {
        SimpleNeuralNetwork nn = new SimpleNeuralNetwork();

        // Datos de entrenamiento para una puerta lógica AND
        double[][] trainingInputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
        double[] trainingOutputs = { 0, 0, 0, 1 };

        // Entrenamiento de la red
        for (int epoch = 0; epoch < 10000; epoch++) {
            for (int i = 0; i < trainingInputs.length; i++) {
                nn.train(trainingInputs[i], trainingOutputs[i]);
            }
        }

        // Prueba de la red
        for (int i = 0; i < trainingInputs.length; i++) {
            double output = nn.forward(trainingInputs[i]);
            System.out.printf("Entrada: %s Salida esperada: %.0f Salida de la red: %.5f%n",
                    java.util.Arrays.toString(trainingInputs[i]), trainingOutputs[i], output);
        }
    }
}

