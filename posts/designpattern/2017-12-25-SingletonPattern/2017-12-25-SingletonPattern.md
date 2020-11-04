---
title: Design Patterns | Creational Design Patterns | Singleton Pattern
description: Singleton Pattern is a creational design pattern which makes sure only a single instance of that class is created and returned each time when a request for an object is made.
pagetitle: Singleton Pattern
summary: Singleton Pattern is a creational design pattern which makes sure only a single instance of that class is created and returned each time when a request for an object is made.
date: '2017-12-25'
update_date: '2019-09-09'
tags:
  - Design Pattern
label:
  - DesignPattern
slug: design-pattern/singleton-pattern
published: true
image: ../../common/atech-guide.png
---

## Characteristics of Singleton 
- Private Constructor
- Private Instance of the class
- Static in nature
- Thread safe
- No parameters to the constructor

## Examples of Singleton
- Instance of Runtime (Runtime.getRuntime())
- Spring beans (Default mode is singleton)

## Java Implementaion

**Singleton.class**{: .heading1}

```java
public class Singleton {

  // Lazy loading
  // Volatile is necessary to avoid threads to see half initialized instance of Singleton
  private static volatile Singleton instance = null;

  private Singleton(){

  }

  /* Synchronizing at method level will synchronize every call to getInstance
   */
  /*public static synchronized Singleton getInstance(){

    if(instance == null){
      instance = new Singleton();
    }
    return instance;
  }*/

  /* We want to synchronize only when we are creating the object to avoid race condition
  Returning the created object should not be blocked
   */
  public static Singleton getInstance(){

    if(instance == null){
      synchronized (Singleton.class){
        // Double checking to make sure thread already inside first null check
        // doesn't create Race Condition
        if(instance == null){
          instance = new Singleton();
        }
      }

    }
    return instance;
  }
}
```

<br/>

**SingletonImpl.class**{: .heading1}

```java
public class SingletonImpl {

  public static void main(String[] args) {

    Singleton instance = Singleton.getInstance();
    System.out.println(instance); // singleton.Singleton@610455d6

    Singleton secondInstance = Singleton.getInstance();
    System.out.println(secondInstance); // singleton.Singleton@610455d6
  }

}
```

## Disadvantages of Singleton
- Over Usage
- Hindrance in writing Unit test
- Confused with Factory design pattern (where we pass an argument to constructor for creating objects)

To access the full working code sample, click [here](https://github.com/atechguide/designpattern-blog/tree/master/creational/src/main/java/singleton "Singleton"){:target="_blank" rel="nofollow noopener"}

## References
- [Singleton Design Pattern | Implementation and Disadvantages | Clean Code Series](https://www.youtube.com/watch?v=bPIRGre9JHY){:target="_blank" rel="nofollow noopener"}
