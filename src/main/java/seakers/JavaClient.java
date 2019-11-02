package seakers;

import org.apache.thrift.TException;
import org.apache.thrift.transport.TSSLTransportFactory;
import org.apache.thrift.transport.TTransport;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TSSLTransportFactory.TSSLTransportParameters;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TProtocol;
import seakers.client.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Collections;

public class JavaClient {
    public static void main(String [] args) {

        if (args.length != 1) {
            System.out.println("Please enter 'simple' or 'secure'");
            System.exit(0);
        }

        try {
            TTransport transport;
            if (args[0].contains("simple")) {
                transport = new TSocket("localhost", 9090);
                transport.open();
            } else {
                /*
                 * Similar to the server, you can use the parameters to setup client parameters or
                 * use the default settings. On the client side, you will need a TrustStore which
                 * contains the trusted certificate along with the public key.
                 * For this example it's a self-signed cert.
                 */
                TSSLTransportParameters params = new TSSLTransportParameters();
                params.setTrustStore("../../lib/java/test/.truststore", "thrift", "SunX509", "JKS");
                /*
                 * Get a client transport instead of a server transport. The connection is opened on
                 * invocation of the factory method, no need to specifically call open()
                 */
                transport = TSSLTransportFactory.getClientSocket("localhost", 9091, 0, params);
            }

            TProtocol protocol = new TBinaryProtocol(transport);
            PythonNeuralNetInterface.Client client = new PythonNeuralNetInterface.Client(protocol);

            perform(client);

            transport.close();
        } catch (TException x) {
            x.printStackTrace();
        }
    }

    private static List<Integer> string2Intlist(String arch)
    {
        List<Integer> arch_Intlist = new ArrayList<>();
        // arch_Intlist =  Collections.nCopies(60, 0);
        for (int i = 0; i < 60; i++) {
            if (arch.substring(i, i + 1).equals("1")) {
                arch_Intlist.add(1);
            } else {
                arch_Intlist.add(0);
            }
        }
        return arch_Intlist;
    }

    private static void perform(PythonNeuralNetInterface.Client client) throws TException
    {
        client.ping();
        System.out.println("ping");

        // String problem = "Climate_Centric";

        // Testing with a test architecture
        String testarch= "111100001100011100000000011100000011100011000000000000000011";
        List<Integer> testarch_list = string2Intlist(testarch);

        try {
            NeuralNetScores scores = client.neuralNetArchitectureEval(testarch_list);

        } catch (TException texcept) {
            System.out.println("Thrift Exception: " + texcept);
            texcept.printStackTrace();
        }

    }

}

