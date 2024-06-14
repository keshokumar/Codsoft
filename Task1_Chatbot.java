import java.util.*;

public class Task1_Chatbot{
    public static void main(String args[]){
        Scanner sc=new Scanner(System.in);
        String s;
        System.out.println("Welcome to Chatbot");
        Response re=new Response();
        while(true){
            s=sc.nextLine();
            if(s.equals("exit")){
                System.out.println("Thank You see you later!");
                break;
            }
            re.fun(s);
        }
    }
}
class Response{
    public void fun(String s){
        System.out.println("User: "+s);
        String all_data=new String();
        all_data="Chatbot: Hello! How can I help you?,There is a virtual Internship available in Codsoft,I am digital source I am always free to chat you!,Today's weather is sunny with a chance of rain.,We offer internships in various domains including software development data science and Artificial Intelligent.,Their office is located in TamilNadu.,";
        String arr[]=all_data.split(",");
        for(String n:arr){
            if(s.equalsIgnoreCase("hello")){
                if(n.startsWith("Chatbot")){
                    System.out.println(n);
                    return;
            }
        }
            else if(s.equalsIgnoreCase("I want do internship")){
                if(n.startsWith("There")){
                    System.out.println("Chatbot: "+n);
                    return;
                }
            }
            else if(s.equalsIgnoreCase("Are you free to chat")){
                if(n.startsWith("I am digital")){
                    System.out.println("Chatbot: "+n);
                    return;
                }
            }
            else if(s.equalsIgnoreCase("what's the weather like")){
                if(n.startsWith("Today")){
                    System.out.println("Chatbot: "+n);
                    return;
                }
            }
            else if(s.equalsIgnoreCase("what internships are available")){
                if(n.startsWith("We offer")){
                    System.out.println("Chatbot: "+n);
                    return;
                }
            }
            else if(s.equalsIgnoreCase("where is codSoft office located")){
                if(n.startsWith("Their office")){
                    System.out.println("Chatbot: "+n);
                    return;
                }
            }
            else {
               break;
            }
            
    } System.out.println("Chatbot: I did not understand that. Can you please clarify?");
}

}
