import java.util.Scanner;

public class TokenRing {
    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.print("Enter the number of processes: ");
        int n = sc.nextInt();

        System.out.print("Enter the process requesting critical section: ");
        int x = sc.nextInt();

        // Optional validation
        if (x < 0 || x >= n) {
            System.out.println("Invalid process number!");
            return;
        }

        int token = 0;

        // Pass token until it reaches requested process
        while (token != x) {
            System.out.println("Token passed from " + token + " to " + ((token + 1) % n));
            token = (token + 1) % n;
        }

        // Critical section execution
        System.out.println("Process " + x + " enters Critical Section");
        System.out.println("Process " + x + " exits Critical Section");

        // Pass token to next process
        token = (token + 1) % n;
        System.out.println("Token is passed to " + token);

        sc.close();
    }
}