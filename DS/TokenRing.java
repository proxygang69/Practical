import java.util.Scanner;

public class TokenRing {
    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of processes: ");
        int n = sc.nextInt();

        System.out.print("Enter process requesting critical section: ");
        int req = sc.nextInt();

        int token = 0;

        while (token != req) {
            System.out.println("Token passed from " + token + " to " + ((token + 1) % n));
            token = (token + 1) % n;
        }

        System.out.println("Process " + req + " ENTERS Critical Section");
        System.out.println("Process " + req + " EXITS Critical Section");

        token = (token + 1) % n;
        System.out.println("Token passed to " + token);
    }
}