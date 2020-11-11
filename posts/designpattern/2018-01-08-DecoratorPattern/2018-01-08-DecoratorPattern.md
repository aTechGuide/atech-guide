---
title: Design Patterns | Structural Design Patterns | Decorator Pattern
description: Decorator Pattern is a structural design pattern which is a combination of inheritance and Composition. It dynamically computes the behaviour of a particular implementation by adding behaviour/functionality to an existing object without affecting other object.
pagetitle: Decorator Pattern
summary: Decorator Pattern is a structural design pattern which combines inheritance and Composition.
date: '2018-01-08'
update_date: '2019-09-09'
tags:
  - Design Pattern
keywords:
  - Design Pattern
label:
  - DesignPattern
slug: design-pattern/decorator-pattern
published: true
image: ../../common/atech-guide.png
---

## Characteristics of Decorator
- Add behaviour/functionality to an existing object without affecting other object 
- Known as Wrappers
- Dynamically computes the behaviour of a particular implementation
- Combination of inheritance and Composition

## Examples of Decorator
- IO Streams

## Java Implementaion

**Phone.class**{: .heading1}  

```java
public interface Phone {
  String build();
}
```

<br/>

**SmartPhone.class**{: .heading1}  

```java
/*
To decorate Phone we need to create an Abstract implementation
Which acts as decorator
 */
public abstract class SmartPhone implements Phone {

  private Phone phone;

  public SmartPhone(Phone phone) {
    this.phone = phone;
  }

  @Override
  public String build(){
    return phone.build();
  }
}
```

<br/>

**AndroidPhone.class**{: .heading1}  

```java
/*
Decorator Implementation
 */
public class AndroidPhone extends SmartPhone {

  public AndroidPhone(Phone phone) {
    super(phone);
  }

  /*
  Adding new functionality to existing implementation
   */
  @Override
  public String build(){
    return super.build() + addOS();
  }

  private String addOS() {
    return " + Android OS v8.0";
  }
}
```

## Disadvantages of Decorator
- New Class for every feature
- Complex because of a large number of objects
- Complex for clients as it requires large chaining of objects

To access the full working code sample, click [here](https://github.com/atechguide/designpattern-blog/tree/master/structural/src/main/java/decorator "Decorator"){:target="_blank" rel="nofollow noopener"}

## References
- [Decorator Design Pattern | Implementation and Disadvantages | Clean Code Series](https://www.youtube.com/watch?v=vqy8BL0xV0c){:target="_blank" rel="nofollow noopener"}
