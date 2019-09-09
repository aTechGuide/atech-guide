---
title: Design Patterns | Structural Design Patterns | Composite Pattern
description: Composite Pattern is a structural design pattern which is used when we need a tree structure to traverse across the object in hierarchical fashion.
pagetitle: Composite Pattern
summary: Composite Pattern is a structural design pattern which is used when we need a tree structure to traverse across the object in hierarchical fashion.
date: '2018-01-02'
update_date: '2019-09-09'
tags:
  - Structural Design Pattern
label:
  - DesignPattern
slug: design-pattern/composite-pattern
published: true
image: ../../common/atech-guide.png
---

## Characteristics of Composite
- When we need a tree structure to traverse across the object in hierarchical fashion
- It can be divided into 3 parts
  - Component: Which is at a higher level, an abstract class/interface e.g. Employee
  - Composite: Implementation of a component, but will also contain children of component e.g. Lead, Manager
  - Leaf: Implementation/Concrete class e.g. Developer

## Java Implementaion

**Employee.class**{: .heading1}  

```java
public abstract class Employee {

  protected String name;
  protected Integer empID;
  protected Long salary;

  public Employee(String name, Integer empID, Long salary) {
    this.name = name;
    this.empID = empID;
    this.salary = salary;
  }

  public void add(Employee employee){
    throw new UnsupportedOperationException("Can't add reportee");
  }

  public void remove(Employee employee){
    throw new UnsupportedOperationException("Can't remove reportee");
  }

  public abstract String toString();
}

```

<br/>

**Developer.class**{: .heading1}  

```java
public class Developer extends Employee {

  public Developer(String name, Integer empID, Long salary) {
    super(name, empID, salary);
  }

  public String toString() {

    StringBuilder builder = new StringBuilder("Developer: ");
    builder.append(name);
    builder.append(",");
    builder.append(empID);

    return builder.toString();
  }
}

```

<br/>

**Lead.class**{: .heading1}  

```java
public class Lead extends Employee {

  private List<Employee> employees = new ArrayList<Employee>();

  public Lead(String name, Integer empID, Long salary) {
    super(name, empID, salary);
  }

  @Override
  public void add(Employee employee){
    employees.add(employee);
  }

  @Override
  public void remove(Employee employee){
    employees.remove(employee);
  }

  @Override
  public String toString() {

    final StringBuilder builder = new StringBuilder("Lead: ");
    builder.append(name);
    builder.append(",");
    builder.append(" " + empID);
    builder.append(", Emloyees: ");
    employees.forEach(employee -> builder.append(" -> " + employee.toString()));

    return builder.toString();
  }
}
```

<br/>

**CompositeExample.class**{: .heading1}  

```java
public class CompositeExample {
/*
Component: Employee
Composite: Lead, Manager
Leaf: Developer
 */

  public static void main(String[] args) {
    Employee developer1 = new Developer("Kamran", 1, 100L);
    Employee developer2 = new Developer("Palash", 2, 100L);
    Employee developer3 = new Developer("Tilak", 3, 100L);
    Employee developer4 = new Developer("Ali", 4, 100L);

    Employee lead1 = new Lead("Mayank", 5, 1000L);
    Employee lead2 = new Lead("Prakhar", 6, 1500L);

    Employee manager = new Manager("Will", 7, 4000L);

    lead1.add(developer1);
    lead1.add(developer2);
    lead2.add(developer3);

    manager.add(lead1);
    manager.add(lead2);
    manager.add(developer4);

    System.out.println(lead1.toString());
    // Lead: Mayank, 5, Emloyees:  -> Developer: Kamran,1 -> Developer: Palash,2
    System.out.println(lead2.toString());
    // Lead: Prakhar, 6, Emloyees:  -> Developer: Tilak,3
    System.out.println(manager.toString());
    // Manager: Will,7, Emloyees:  -> Lead: Mayank, 5, Emloyees:  -> Developer: Kamran,1 -> Developer: Palash,2
    // -> Lead: Prakhar, 6, Emloyees:  -> Developer: Tilak,3 -> Developer: Ali,4

  }
}
```

## Disadvantages of Composite
- Duplication of data (in our example we have created two lists)
- Overly simple, So people may tend to create more and more composite classes without giving due consideration

To access the full working code sample, click [here](https://github.com/kamranalinitb/designpattern-blog/tree/master/structural/src/main/java/composite "Composite"){:target="_blank" rel="nofollow noopener"}

## References
- [Composite Design Pattern | Implementation and Disadvantages | Clean Code Series](https://www.youtube.com/watch?v=3wmXiuZFReU){:target="_blank" rel="nofollow noopener"}
