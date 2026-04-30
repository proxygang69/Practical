

import java.rmi.*;
import java.rmi.server.*;

public class StringOpImpl extends UnicastRemoteObject implements StringOp {

    public StringOpImpl() throws RemoteException {
        super();
    }

    public String compare(String a, String b) throws RemoteException {
        if (a.compareTo(b) > 0)
            return a;
        else
            return b;
    }
}

