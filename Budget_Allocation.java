import java.io.*;
import static java.lang.System.*;
import java.util.ArrayList;
import java.util.Scanner;
import static java.lang.Math.*;



class Budget_Allocation {
	public static void main(String str[]) throws IOException {
        // double x & y is the percent likelihood that a person will 
        // get a deposit (determined by part 1) 
        
        double x = 0.3;
        double y = 0.5;
        double z = 0.5;

        double allocation_X = (Math.sin(x * Math.PI)) / ((Math.sin(x * Math.PI)) + (Math.sin(y * Math.PI)) + (Math.sin(z * Math.PI)));
        double allocation_Y = (Math.sin(y * Math.PI)) / ((Math.sin(x * Math.PI)) + (Math.sin(y * Math.PI)) + (Math.sin(z * Math.PI)));
        double allocation_Z = (Math.sin(z * Math.PI)) / ((Math.sin(x * Math.PI)) + (Math.sin(y * Math.PI)) + (Math.sin(z * Math.PI)));


        System.out.println(allocation_X);
        System.out.println(allocation_Y);
        System.out.println(allocation_Z);

        System.out.println(allocation_Z + allocation_Y + allocation_X);
    }
}