---
title: Design Patterns | Creational Design Patterns | Factory Pattern
description: Factory Pattern is a creational design pattern which doesn't expose instantiation or creation logic and returns new instace when object is asked.
pagetitle: Factory Pattern
summary: Factory Pattern is a creational design pattern which doesn't expose instantiation or creation logic and returns new instace when object is asked.
date: '2017-12-27'
update_date: '2019-09-09'
tags:
  - Design Pattern
label:
  - DesignPattern
slug: design-pattern/factory-pattern
published: true
image: ../../common/atech-guide.png
---

## Characteristics of Factory 
- Doesn't expose instantiation or creation logic
- New instances are returned each time
- Sub classes create objects
- Implemented using Abstract class/Interface
- Factory design pattern is opposite to [Singleton pattern]({% post_url 2017-12-25-SingletonPattern %}){:target="_blank"} because in Singleton there is only one instance that is returned but in the factory we return a new instance 

## Examples of Factory
- Calendar (Calendar.getInstance().get(Calendar.DAY_OF_WEEK))
- NumberFormat(NumberFormat.getInstance())

## Java Implementaion

**PhoneFactory.class**{: .heading1}  

```java
public class PhoneFactory {

  public static Phone getPhone(PhoneType phoneType){

    switch (phoneType) {
      case ANDOID:
        return new AndroidPhone();
      case IPHONE:
        return new IPhone();
        default:
          return null;
    }

  }
}
```

<br/>

**FactoryExample.class**{: .heading1}  

```java
public class FactoryExample {

  public static void main(String[] args) {
    Phone androidPhone = PhoneFactory.getPhone(PhoneType.ANDOID);
    Phone iPhone = PhoneFactory.getPhone(PhoneType.IPHONE);

    System.out.println(androidPhone);
    System.out.println(iPhone);
  }
}
```

## Disadvantages of Factory
- Complex (Need lots of classes/code)
- Creation in subclass (which abstracts lot of things as we don't know what is happening)
- Refactoring is difficult (Too much code to modify)
 
To access the full working code sample, click [here](https://github.com/kamranalinitb/designpattern-blog/tree/master/creational/src/main/java/factory "Factory"){:target="_blank" rel="nofollow noopener"}

## References
- [Factory Design Pattern | Implementation and Disadvantages | Clean Code Series](https://www.youtube.com/watch?v=a46oBUV8mZ4 "Factory Design Pattern"){:target="_blank" rel="nofollow noopener"}
