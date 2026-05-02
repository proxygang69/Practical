/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Main.java to edit this template
 */
package client;
import calculator.CalciService;
import calculator.CalciService_Service;

public class Client {

    public static void main(String[] args) {

        try {
            // Create service object (generated class)
            CalciService_Service service = new CalciService_Service();

            // Get the port (proxy to the web service)
            CalciService port = service.getCalciServicePort();

            double a = 15;
            double b = 5;

            // Call web service methods
            System.out.println("Addition: " + port.add(a, b));
            System.out.println("Subtraction: " + port.subtract(a, b));
            System.out.println("Multiplication: " + port.multiply(a, b));
            System.out.println("Division: " + port.divide(a, b));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}