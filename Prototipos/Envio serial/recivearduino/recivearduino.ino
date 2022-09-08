char incomingByte = 0;
char x;
void setup() {

Serial.begin(9600); // opens serial port, sets data rate to 9600 bps
Serial.setTimeout(1);
Serial.println("Ready!");
pinMode(2, OUTPUT); 
pinMode(3, OUTPUT);
pinMode(4, OUTPUT);
pinMode(5, OUTPUT);
pinMode(6, OUTPUT);
pinMode(7, OUTPUT);
pinMode(8, OUTPUT);
pinMode(9, OUTPUT);
digitalWrite(2, 0);  // start with the "dot" off
delay(1000);
}

void loop() {

while (!Serial.available());
incomingByte = Serial.read(); // read the incoming byte:

Serial.print(" I received:");

Serial.println(incomingByte);
int a=incomingByte;
delay(200);  
if (a==1) 
digitalWrite(2,LOW);
digitalWrite(3,HIGH);
digitalWrite(4,LOW);
digitalWrite(5,LOW);
digitalWrite(6,HIGH);
digitalWrite(7,LOW);
digitalWrite(8,LOW);
digitalWrite(9,LOW);
if (a==2)
digitalWrite(2,LOW);
digitalWrite(3,LOW);
digitalWrite(4,HIGH);
digitalWrite(5,HIGH);
digitalWrite(6,HIGH);
digitalWrite(7,HIGH);
digitalWrite(8,LOW);
digitalWrite(9,HIGH);



}
