---
title: Design Patterns | Structural Design Patterns | Adapter Pattern
description: Adapter Pattern is a structural design pattern which bridges gap between two interfaces. It is used when we write new interface (new client) which will work with legacy code keeping in mind we don't end up adding new functionality to existing functionality.
pagetitle: Adapter Pattern
summary: Adapter Pattern is a structural design pattern which bridges gap between two interfaces.
date: '2018-01-02'
update_date: '2019-09-09'
tags:
  - Design Pattern
label:
  - DesignPattern
slug: design-pattern/adapter-pattern
published: true
image: ../../common/atech-guide.png
---

## Characteristics of Adapter 
- It bridges gap between two interfaces
- It is used when we write new interface (new client) which will work with legacy code
- It is used when without touching the legacy code, we want to
  - convert data type
  - conversion of form1 to form2
- **Caveat** : Don't add new functionality to existing functionality else it will become the decorator design pattern

## Examples of Adapter
- In Collection API, Arrays were implemented first and later List was implemented so in order to convert Array into List we have Arrays.asList()
- Adapters for Input/Output streams

## Java Implementaion

**AndroidToIphoneAdapter.class**{: .heading1}  

```java
/*
Charge Android phone with IPhone charger.
We are passing Android phone into it but it can be charged using an iphone charger.
 */
public class AndroidToIphoneAdapter implements IPhone{

  private AndroidPhone androidPhone;

  public AndroidToIphoneAdapter(AndroidPhone androidPhone) {
    this.androidPhone = androidPhone;
  }

  public void charge() {
    androidPhone.charge();
  }
}

```

<br/>

**AdapterExample.class**{: .heading1}  

```java
public class AdapterExample {

  public static void main(String[] args) {

    AndroidCharger androidCharger = new AndroidCharger();
    AndroidPhone androidPhone = new Pixel5Phone();
    androidCharger.charge(androidPhone);

    IPhoneCharger iPhoneCharger = new IPhoneCharger();
    IPhone iPhone = new IPhone10();
    iPhoneCharger.charge(iPhone);

    /*
    We will pass Android phone to AndroidToIphoneAdapter and use Iphone Charger to charge Android
    Phone
     */
    AndroidToIphoneAdapter adapter = new AndroidToIphoneAdapter(androidPhone);
    iPhoneCharger.charge(adapter);
  }
}
```

## Disadvantages of Adapter
- Multiple adapters may add to the confusion

To access the full working code sample, click [here](https://github.com/atechguide/designpattern-blog/tree/master/structural/src/main/java/adapter "Adapter"){:target="_blank" rel="nofollow noopener"}

## References
- [Adapter Design Pattern | Implementation and Disadvantages | Clean Code Series](https://www.youtube.com/watch?v=hbXHzweWKMU){:target="_blank" rel="nofollow noopener"}