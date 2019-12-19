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

import java.io.*;
import java.util.*;

public class CCNeuralNetDatasetGeneration {

    public static void main(String[] args) throws IOException {

        // PATH
        String resourcesPath = "C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_resources-master";
        String architectureCsvPath = "C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_AMOS_V1\\instrument_numbers.csv";

        // Initialize required classes for VASSAR
        AssigningParams params = new ClimateCentricParams(resourcesPath, "CRISP-ATTRIBUTES", "test", "normal");
        AbstractArchitectureEvaluator eval = new ArchitectureEvaluator();
        ArchitectureEvaluationManager AE = new ArchitectureEvaluationManager(params, eval);
        AbstractArchitecture testArch;
        AE.init(4);
        OrekitConfig.init(4, params.orekitResourcesPath);

        // String variable determining the type of architectures generated
        // "MedInstr" - Booleans generated are pseudorandom (biased towards 0)
        // "LowInstr" - Booleans generated are uniformly random
        String DatasetType = "MedInstr";

        // Reading the number of architectures for each number of instruments from instrument_numbers.csv and saving them
        // to an array

        List<List<String>> lines = new ArrayList<>();
        BufferedReader csvReader = new BufferedReader(new FileReader(architectureCsvPath));

        int row_count = 0;
        String line;
        boolean header = true;

        while((line = csvReader.readLine()) != null) {
            if (header) {
               header = false;
               continue;
            }
            String[] values = line.split(",");
            lines.add(Arrays.asList(values));
            row_count++;
        }
        //System.out.println(lines);
        int[][] ArchNumbersPerInstrument = new int[60][row_count];
        int[] MedInstrumentNums = new int[60];
        int[] LowInstrumentNums = new int[60];
        int lineNum = 0;
        for(List<String> line_current: lines) {
            int columnNum = 0;
            for (String value : line_current) {
                ArchNumbersPerInstrument[lineNum][columnNum] = Integer.parseInt(value);
                columnNum++;
            }
            lineNum++;
        }
        for (int n = 0; n < 59; n++) {
            MedInstrumentNums[n] = ArchNumbersPerInstrument[n][0];
            LowInstrumentNums[n] = ArchNumbersPerInstrument[n][1];
        }
        //System.out.println(Arrays.toString(MedInstrumentNums));
        //System.out.println(Arrays.toString(LowInstrumentNums));

        //int n_archs = 15000; // Setting no. of architectures to create as 15,000
        //String[] archs = new String[n_archs]; // Initialize array of architectures

        switch (DatasetType) {

            case "MedInstr": {
                File csvFile = new File("vassar_data_medinstr.csv");
                FileWriter csvWrite = new FileWriter(csvFile);
                csvWrite.append("Architecture");
                csvWrite.append(",");
                csvWrite.append("Science Benefit");
                csvWrite.append(",");
                csvWrite.append("Cost");
                csvWrite.append("\n");

                // Generate specific numbers of architectures with each number of instruments to represent a Gaussian Distribution
                int[] numArchitecturesPerInstrument = new int[59];
                numArchitecturesPerInstrument = MedInstrumentNums;
                //Arrays.fill(numArchitecturesPerInstrument, 1); // For testing
                // ArchitectureArray = new String[n_archs];
                HashSet<String> Architectures = new HashSet<String>();
                Architectures.add("000000000000000000000000000000000000000000000000000000000000");

                for (int i = 0; i < 59; i++) {
                    System.out.println("Adding Architecture(s) with " + Integer.toString(i+1) + " Instrument(s)");
                    int ArchsPerInstr = numArchitecturesPerInstrument[i];
                    String[] ArchArray;
                    ArchArray = GenerateArchs(i+1, ArchsPerInstr);
                    /*
                    for (int j = 0; j < ArchsPerInstr; j++) {
                        Architectures.add(ArchArray[j]);
                    }
                    */
                    Architectures.addAll(Arrays.asList(ArchArray).subList(0, ArchsPerInstr));
                    System.out.println("Architecture(s) Added");
                }
                Architectures.add("111111111111111111111111111111111111111111111111111111111111");

                System.out.println("Evaluating Architectures.....");
                int count = 0;
                for (String m: Architectures) {
                    String currentArchitecture = m;
                    testArch = new Architecture(currentArchitecture, 1, params);
                    Result result = AE.evaluateArchitectureSync(testArch, "Slow");
                    result.cleanExtras();
                    csvWrite.append(testArch.toString(""));
                    csvWrite.append(",");
                    csvWrite.append(Double.toString(result.getScience()));
                    csvWrite.append(",");
                    csvWrite.append(Double.toString(result.getCost()));
                    csvWrite.append("\n");

                    count = count + 1;
                    System.out.println(Integer.toString(count));
                    if (count % 100 == 0) {
                        AE.getResourcePool().poolClean();
                        System.out.println("Rete Clean initiatied");
                    }
                }
                //Iterator<String> iter = Architectures.iterator();
                csvWrite.flush();
                csvWrite.close();
                break;
            }
            case "LowInstr": {
                File csvFile = new File("vassar_data_lowinstr.csv");
                FileWriter csvWrite = new FileWriter(csvFile);
                csvWrite.append("Architecture");
                csvWrite.append(",");
                csvWrite.append("Science Benefit");
                csvWrite.append(",");
                csvWrite.append("Cost");
                csvWrite.append("\n");

                // Generate specific numbers of architectures with each number of instruments to represent a Gaussian Distribution
                int[] numArchitecturesPerInstrument = new int[59];
                numArchitecturesPerInstrument = LowInstrumentNums;
                //Arrays.fill(numArchitecturesPerInstrument, 1); // For testing
                // ArchitectureArray = new String[n_archs];
                HashSet<String> Architectures = new HashSet<String>();
                Architectures.add("000000000000000000000000000000000000000000000000000000000000");

                for (int i = 0; i < 59; i++) {
                    //System.out.println("Adding Architecture(s) with " + Integer.toString(i+1) + " Instrument(s)");
                    int ArchsPerInstr = numArchitecturesPerInstrument[i];
                    String[] ArchArray;
                    ArchArray = GenerateArchs(i+1, ArchsPerInstr);
                    /*
                    for (int j = 0; j < ArchsPerInstr; j++) {
                        Architectures.add(ArchArray[j]);
                    }
                    */
                    Architectures.addAll(Arrays.asList(ArchArray).subList(0, ArchsPerInstr));
                    System.out.println("Architecture(s) Added");
                }
                Architectures.add("111111111111111111111111111111111111111111111111111111111111");

                System.out.println("Evaluating Architectures.....");
                int count = 0;
                for (String m: Architectures) {
                    String currentArchitecture = m;
                    testArch = new Architecture(currentArchitecture, 1, params);
                    Result result = AE.evaluateArchitectureSync(testArch, "Slow");
                    result.cleanExtras();
                    csvWrite.append(testArch.toString(""));
                    csvWrite.append(",");
                    csvWrite.append(Double.toString(result.getScience()));
                    csvWrite.append(",");
                    csvWrite.append(Double.toString(result.getCost()));
                    csvWrite.append("\n");

                    count = count + 1;
                    System.out.println(Integer.toString(count));
                    if (count % 100 == 0) {
                        AE.getResourcePool().poolClean();
                        System.out.println("Rete Clean initiatied");
                    }
                }
                //Iterator<String> iter = Architectures.iterator();
                csvWrite.flush();
                csvWrite.close();
                break;
            }
        }
        OrekitConfig.end();
        AE.clear();
        System.out.println("DONE");
    }

    public static String[] GenerateArchs(int n_Instr, int n_Arch_Instr) {
        System.out.println(Integer.toString(n_Arch_Instr) + " Architecture(s) with " + Integer.toString(n_Instr) + " Instrument(s) being generated");
        String[] architectureList = new String[n_Arch_Instr];
        HashSet<String> Archs_Instr = new HashSet<String>();
        int i = 0;
        //architectureArray = new int [n_Arch_Instr][60];
        while(i < n_Arch_Instr) {
            // onePosition = new int[n_Instr];
            int[] architectureWithInstrument = new int[60];
            HashSet<Integer> Ones = new HashSet<Integer>();
            int j = 0;
            while (j < n_Instr) {
                Random rand = new Random();
                int nextOne = rand.nextInt(60); // generates random integer between 0 and 59
                if (!Ones.contains(nextOne)) {
                    Ones.add(nextOne);
                    j++;
                    // onePosition[j] = nextOne;
                }
            }
            /* for (int k = 0; k < numInstruments; k++) {
                architectureWithInstrument[onePosition[k]] = 1;
            }*/

            for (Integer k: Ones){
                //System.out.println(Integer.toString(k));
                architectureWithInstrument[k] = 1;
            }
            /*for (int m = 0; m < 60; m++) {
                architectureArray[i][m] = architectureWithInstrument[m];
            }
            */
            String ArchString = IntArray2String(architectureWithInstrument);
            if (!Archs_Instr.contains(ArchString)) {
                Archs_Instr.add(ArchString);
                architectureList[i] = ArchString;
                //System.out.println(architectureList[i]);
                i++;
                //System.out.println("Architecture added");
            }
        }
        return architectureList;
    }

    public static String IntArray2String (int[] ArchitectureArray) {
        StringBuilder ArchitectureSB = new StringBuilder();
        for (int i = 0; i < 60; i++) {
            ArchitectureSB.append(Integer.toString(ArchitectureArray[i]));
        }
        String ArchitectureString = ArchitectureSB.toString();
        return ArchitectureString;
    }

}
