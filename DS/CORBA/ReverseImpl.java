import ReverseModule.*;

public class ReverseImpl extends ReversePOA {
    public String reverseString(String input) {
        return new StringBuilder(input).reverse().toString();
    }
}