---
title: Design Patterns | Creational Design Patterns | Abstract Factory Pattern
description: Abstract Factory Pattern is a creational design pattern which is also known as Factory of factories.
pagetitle: Abstract Factory Pattern
summary: Abstract Factory Pattern is a creational design pattern which is also known as Factory of factories.
date: '2017-12-29'
update_date: '2019-09-09'
tags:
  - Design Pattern
label:
  - DesignPattern
slug: design-pattern/abstract-factory-pattern
published: true
image: ../../common/atech-guide.png
---

## Characteristics of Abstract Factory 
- Factory of factories / Collection of factories
- Common abstract class/Interface
- Subclasses create the objects

## Examples of Abstract Factory
- DocumentBuilderFactory

## Java Implementation

**AbstractPhoneFactory.class**{: .heading1}  

```java
public abstract class AbstractPhoneFactory {

  public static OSFactory getFactory(OSType osType){

    switch (osType) {
      case ANDROID:
        return new AndroidFactory();
      case WINDOWS:
        return new WindowsFactory();
      default:
        return null;
    }
  }

}
```

<br/>

**AbstractFactoryExample.class**{: .heading1}  

```java
public class AbstractFactoryExample {

  public static void main(String[] args) {

    OSFactory factory = AbstractPhoneFactory.getFactory(OSType.ANDROID);
    Phone pixelPhone = factory.create(ManufacturerType.GOOGLE);
    pixelPhone.display();

    OSFactory wfactory = AbstractPhoneFactory.getFactory(OSType.WINDOWS);
    Phone nokiaPhone = wfactory.create(ManufacturerType.MICROSOFT);
    nokiaPhone.display();

  }
}
```

## Disadvantages of Abstract Factory
- Complex code
- We realise the need of it while refactoring Facotry Pattern
- It needs knowledge of Factory Pattern
 
To access the full working code sample, click [here](https://github.com/kamranalinitb/designpattern-blog/tree/master/creational/src/main/java/abstractfactory "AbstractFactory"){:target="_blank" rel="nofollow noopener"}

## References
- [Abstract Factory Design Pattern | Implementation and Disadvantages | Clean Code Series](https://www.youtube.com/watch?v=hWVfRwgfdGg){:target="_blank" rel="nofollow noopener"}
