import org.springframework.web.bind.annotation.*;

@RestController
public class MyController {

    // 1. HELLO SERVICE
    @GetMapping("/hello")
    public String hello(@RequestParam String name) {
        return "Hello " + name;
    }

    // 2. CALCULATOR (ADD)
    @GetMapping("/add")
    public int add(@RequestParam int a, @RequestParam int b) {
        return a + b;
    }

    // 3. SIMPLE INTEREST
    @GetMapping("/interest")
    public double interest(@RequestParam double p,
                           @RequestParam double r,
                           @RequestParam double t) {
        return (p * r * t) / 100;
    }

    // 4. MILES TO KM
    @GetMapping("/convert")
    public double convert(@RequestParam double miles) {
        return miles * 1.609;
    }
}