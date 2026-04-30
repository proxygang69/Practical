

import java.rmi.*;

public interface StringOp extends Remote {
    String compare(String a, String b) throws RemoteException;
}

