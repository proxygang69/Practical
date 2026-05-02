package Calculator;

import javax.jws.WebService;
import javax.jws.WebMethod;
import javax.jws.WebParam;

@WebService(serviceName = "CalciService")
public class CalciService {

    @WebMethod(operationName = "add")
    public double add(
        @WebParam(name = "a") double a,
        @WebParam(name = "b") double b) {
        return a + b;
    }

    @WebMethod(operationName = "subtract")
    public double subtract(
        @WebParam(name = "a") double a,
        @WebParam(name = "b") double b) {
        return a - b;
    }

    @WebMethod(operationName = "multiply")
    public double multiply(
        @WebParam(name = "a") double a,
        @WebParam(name = "b") double b) {
        return a * b;
    }

    @WebMethod(operationName = "divide")
    public double divide(
        @WebParam(name = "a") double a,
        @WebParam(name = "b") double b) {
        if (b == 0) {
            throw new ArithmeticException("Cannot divide by zero");
        }
        return a / b;
    }
}