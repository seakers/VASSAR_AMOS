package seakers;

import seakers.orekit.util.OrekitConfig;
import seakers.vassar.Result;
import seakers.vassar.architecture.AbstractArchitecture;
import seakers.vassar.evaluation.AbstractArchitectureEvaluator;
import seakers.vassar.evaluation.ArchitectureEvaluationManager;
import seakers.vassar.problems.Assigning.Architecture;
import seakers.vassar.problems.Assigning.ArchitectureEvaluator;
import seakers.vassar.problems.Assigning.AssigningParams;
import seakers.vassar.problems.Assigning.ClimateCentricParams;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class CCGenerateAndEvaluateArchs {

    public static void main(String[] args) throws IOException {

        // PATH
        String resourcesPath = "C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_resources-master";

        // Initialize required classes for VASSAR
        AssigningParams params = new ClimateCentricParams(resourcesPath, "CRISP-ATTRIBUTES", "test", "normal");
        AbstractArchitectureEvaluator eval = new ArchitectureEvaluator();
        ArchitectureEvaluationManager AE = new ArchitectureEvaluationManager(params, eval);
        AbstractArchitecture testArch;
        AE.init(4);
        OrekitConfig.init(4, params.orekitResourcesPath);

        // String variable determining the type of architectures generated
        // "LessArchs" - Booleans generated are pseudorandom (biased towards 0)
        // "Uniform" - Booleans generated are uniformly random
        String BoolSampling = "LessArchs";

        int n_archs = 10000; // Setting no. of architectures to create as 10,000
        String[] archs = new String[n_archs]; // Initialize array of architectures

        switch (BoolSampling) {

            case "Uniform":
                File csvFile = new File("vassar_data_uniform.csv");
                FileWriter csvWrite = new FileWriter(csvFile);
                csvWrite.append("Architecture");
                csvWrite.append(",");
                csvWrite.append("Science Benefit");
                csvWrite.append(",");
                csvWrite.append("Cost");
                csvWrite.append("\n");

                // Populate array of architectures
                Random rand = new Random();
                boolean val;
                boolean duplicate = false;
                for (int i = 0; i < n_archs; i++) {
                    StringBuilder current_arch = new StringBuilder();
                    for (int j = 0; j < 60; j++) {
                        val = rand.nextBoolean();
                        if (val) {
                            current_arch.append(Integer.toString(1));
                        } else {
                            current_arch.append(Integer.toString(0));
                        }
                    }
                    archs[i] = current_arch.toString();
                    // duplicate = false;
                    for (int j = 0; j < i; j++) {
                        int same = current_arch.toString().compareTo(archs[j]);
                        if (same == 0) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (duplicate) {
                        i--;
                        continue;
                    }
                    testArch = new Architecture(current_arch.toString(), 1, params);
                    Result result = AE.evaluateArchitectureSync(testArch, "Slow");
                    result.cleanExtras();
                    csvWrite.append(testArch.toString(""));
                    csvWrite.append(",");
                    csvWrite.append(Double.toString(result.getScience()));
                    csvWrite.append(",");
                    csvWrite.append(Double.toString(result.getCost()));
                    csvWrite.append("\n");

                    if (i % 100 == 0) {
                        AE.getResourcePool().poolClean();
                        System.out.println("Rete Clean initiatied");
                    }

                    System.out.println(Integer.toString(i));
                }
                csvWrite.flush();
                csvWrite.close();
                break;

            case "LessArchs":
                csvFile = new File("vassar_data_lessarchs.csv");
                csvWrite = new FileWriter(csvFile);
                csvWrite.append("Architecture");
                csvWrite.append(",");
                csvWrite.append("Science Benefit");
                csvWrite.append(",");
                csvWrite.append("Cost");
                csvWrite.append("\n");

                // Populate array of architectures
                rand = new Random();
                double TrueProb = 0.2;
                // boolean val;
                duplicate = false;
                for (int i = 0; i < n_archs; i++) {
                    StringBuilder current_arch = new StringBuilder();
                    for (int j = 0; j < 60; j++) {
                        val = rand.nextFloat() < TrueProb;
                        if (val) {
                            current_arch.append(Integer.toString(1));
                        } else {
                            current_arch.append(Integer.toString(0));
                        }
                    }
                    archs[i] = current_arch.toString();
                    // duplicate = false;
                    for (int j = 0; j < i; j++) {
                        int same = current_arch.toString().compareTo(archs[j]);
                        if (same == 0) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (duplicate) {
                        i--;
                        continue;
                    }
                    testArch = new Architecture(current_arch.toString(), 1, params);
                    Result result = AE.evaluateArchitectureSync(testArch, "Slow");
                    result.cleanExtras();
                    csvWrite.append(testArch.toString(""));
                    csvWrite.append(",");
                    csvWrite.append(Double.toString(result.getScience()));
                    csvWrite.append(",");
                    csvWrite.append(Double.toString(result.getCost()));
                    csvWrite.append("\n");

                    if (i % 100 == 0) {
                        AE.getResourcePool().poolClean();
                        System.out.println("Rete Clean initiatied");
                    }

                    System.out.println(Integer.toString(i));
                }
                csvWrite.flush();
                csvWrite.close();
                break;

        }

        OrekitConfig.end();
        AE.clear();
        System.out.println("DONE");
    }
}
