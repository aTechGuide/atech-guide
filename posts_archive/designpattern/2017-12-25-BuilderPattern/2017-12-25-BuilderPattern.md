---
title: Design Patterns | Creational Design Patterns | Builder Pattern
description: Builder Pattern is a creational design pattern which is used to create immutable objects with default values for attributes.
pagetitle: Builder Pattern
summary: Builder Pattern is a creational design pattern which is used to create immutable objects with default values for attributes.
date: '2017-12-25'
update_date: '2019-09-09'
tags:
  - Design Pattern
keywords:
  - Design Pattern
label:
  - DesignPattern
slug: design-pattern/builder-pattern
published: true
image: ../../common/atech-guide.png
---

## Characteristics of Builder 
- Creates immutable objects
- Has static inner class (builder class)
- Provide default values for attributes (if necessary and not provided)
- Solves multiple constructor problem (telescoping constructors)
- Used when we have complex setters and lots of arguments
- Removes the need for setters

## Examples of Builder
- StringBuffer/StringBuilders
- DocumentBuilder
- MockMvcBuilder

## Java Implementaion

**Phone.class**{: .heading1}

```java
public class Phone {

  private String backPanel;
  private String frontPanel;
  private String processor;
  private String camera;

  public Phone(Builder builder) {
    this.frontPanel = builder.frontPanel;
    this.backPanel = builder.backPanel;
    this.processor = builder.processor;
    this.camera = builder.camera;
  }

  public String getBackPanel() {
    return backPanel;
  }

  public String getFrontPanel() {
    return frontPanel;
  }

  public String getProcessor() {
    return processor;
  }

  public String getCamera() {
    return camera;
  }

  public static class Builder {

    private String backPanel;
    private String frontPanel;
    private String processor;
    private String camera;

    public Builder(){

      // Provide default configurations here

    }

    public Phone build(){
      return new Phone(this);
    }

    public Builder frontPanel(String frontPanel){
      this.frontPanel = frontPanel;
      return this;
    }

    public Builder backPanel(String backPanel){
      this.backPanel = backPanel;
      return this;
    }

    public Builder processor(String processor){
      this.processor = processor;
      return this;
    }

    public Builder camera(String camera){
      this.camera = camera;
      return this;
    }
  }
}
```

<br/>

**BuilderExample.class**{: .heading1}

```java
public class BuilderExample {

  public static void main(String[] args) {

    Phone.Builder builder = new Phone.Builder()
        .backPanel("Sandstone")
        .frontPanel("AMOLED")
        .camera("12MP")
        .processor("SnapDragon");

    Phone phone = builder.build();

  }
}
```

## Disadvantages of Builder
- Immutability (Can't change object once created)
- Contains Inner Static class
- Complex 
 
To access the full working code sample, click [here](https://github.com/atechguide/designpattern-blog/tree/master/creational/src/main/java/builder "Builder"){:target="_blank" rel="nofollow noopener"}

## References
- [Builder Design Pattern | Implementation and Disadvantages | Clean Code Series](https://www.youtube.com/watch?v=YmEVYvELt28){:target="_blank" rel="nofollow noopener"}
